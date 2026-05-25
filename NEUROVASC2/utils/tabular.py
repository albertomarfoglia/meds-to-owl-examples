import polars as pl
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def build_features(df: pl.DataFrame) -> pl.DataFrame:

    # Ensure deterministic ordering for "last"
    df = df.sort(["subject_id", "time"], nulls_last=True)

    # ----------------------------
    # 1. Detect event types
    # ----------------------------
    static_codes = df.filter(
        pl.col("time").is_null() & pl.col("numeric_value").is_null()
    )

    static_numeric = df.filter(
        pl.col("time").is_null() & pl.col("numeric_value").is_not_null()
    )

    dynamic_codes = df.filter(
        pl.col("time").is_not_null() & pl.col("numeric_value").is_null()
    )

    dynamic_numeric = df.filter(
        pl.col("time").is_not_null() & pl.col("numeric_value").is_not_null()
    )

    def clean_xgb_feature_name(n):
        n = str(n)
        n = re.sub(r"[\[\]<>]", "", n)  # remove forbidden chars
        # n = re.sub(r"[^0-9a-zA-Z_]+", "_", n)  # replace other specials
        # n = re.sub(r"_+", "_", n)              # collapse multiple _
        n = n.strip("_")
        return n

    # ----------------------------
    # 2. STATIC CODES → categorical
    # (e.g. GENDER//F → gender=F)
    # ----------------------------
    static_cat = (
        static_codes.with_columns(
            [
                pl.col("code").str.split("//").list.get(0).alias("feature"),
                pl.col("code").str.split("//").list.get(-1).alias("value"),
            ]
        )
        .group_by(["subject_id", "feature"])
        .agg(pl.first("value"))
        .pivot(index="subject_id", on="feature", values="value")
    )

    # ----------------------------
    # 3. STATIC NUMERIC → last value
    # ----------------------------
    static_num = (
        static_numeric.group_by(["subject_id", "code"])
        .agg(pl.last("numeric_value").alias("value"))
        .pivot(index="subject_id", on="code", values="value")
        .rename(lambda c: c if c == "subject_id" else f"{clean_xgb_feature_name(c)}")
    )

    # ----------------------------
    # 4. DYNAMIC CODES → counts
    # ----------------------------
    dyn_code = (
        dynamic_codes.group_by(["subject_id", "code"])
        .agg(pl.len().alias("count"))
        .pivot(index="subject_id", on="code", values="count")
        .rename(
            lambda c: c if c == "subject_id" else f"{clean_xgb_feature_name(c)}_count"
        )
        .fill_null(0)
    )

    # ----------------------------
    # 5. DYNAMIC NUMERIC → mean
    # ----------------------------
    dyn_num = (
        dynamic_numeric.group_by(["subject_id", "code"])
        .agg(pl.mean("numeric_value").alias("mean"))
        .pivot(index="subject_id", on="code", values="mean")
        .rename(lambda c: c if c == "subject_id" else f"{clean_xgb_feature_name(c)}")
    )

    # ----------------------------
    # 6. Merge safely
    # ----------------------------
    dfs = [static_cat, static_num, dyn_code, dyn_num]

    final_df = None
    for d in dfs:
        if d is None or d.is_empty():
            continue
        if final_df is None:
            final_df = d
        else:
            final_df = final_df.join(d, on="subject_id", how="full", coalesce=True)

    return final_df # type: ignore


# ------------------------ Metrics ------------------------ #
def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """Compute accuracy, precision, recall, F1, and AUC."""

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # AUC
    if num_classes == 2:
        auc_class = roc_auc_score(y_true, y_prob[:, 1])
        auc_macro = auc_class
        auc_weighted = auc_class

        ap_class = average_precision_score(y_true, y_prob[:, 1])
        ap_macro = ap_class
        ap_weighted = ap_class
    else:
        # multi-class
        auc_class = roc_auc_score(y_true, y_prob, average=None, multi_class="ovr")
        auc_macro = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(
            y_true, y_prob, average="weighted", multi_class="ovr"
        )

        ap_class = average_precision_score(y_true, y_prob, average=None)
        ap_macro = average_precision_score(y_true, y_prob, average="macro")
        ap_weighted = average_precision_score(y_true, y_prob, average="weighted")

    # Precision, recall, F1
    precision_class, recall_class, fscore_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, fscore_weighted, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    metrics = {
        "accuracy": accuracy,
        "auc_class": auc_class,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "ap_class": ap_class,
        "ap_macro": ap_macro,
        "ap_weighted": ap_weighted,
        "precision_class": precision_class,
        "recall_class": recall_class,
        "fscore_class": fscore_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "fscore_macro": fscore_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "fscore_weighted": fscore_weighted,
    }

    return metrics


# ------------------------ Confusion Matrix ------------------------ #
def save_confusion_matrix(y_true, y_pred, result_path, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig = disp.plot().figure_
    fig.savefig(result_path, dpi=600)
    plt.close(fig)


def mean_std_metrics(
    metrics_mean: pd.DataFrame, metrics_std: pd.DataFrame, classes: list[str], digits=2
) -> pd.DataFrame:
    headers = classes + ["MACRO", "WEIGHTED"]

    metrics_mean = metrics_mean.reindex(headers)
    metrics_std = metrics_std.reindex(headers)

    def mean_std_str(mean, std, decimals=digits):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    f1_line = [
        mean_std_str(m, s)
        for m, s in zip(metrics_mean["F1SCORE"], metrics_std["F1SCORE"])
    ]

    f1_line.extend(
        [
            mean_std_str(
                metrics_mean.loc["WEIGHTED", "ACCURACY"],
                metrics_std.loc["WEIGHTED", "ACCURACY"],
            ),
            mean_std_str(
                metrics_mean.loc["WEIGHTED", "AUC"], metrics_std.loc["WEIGHTED", "AUC"]
            ),
            mean_std_str(
                metrics_mean.loc["WEIGHTED", "AP"], metrics_std.loc["WEIGHTED", "AP"]
            ),
        ]
    )

    return pd.DataFrame([f1_line], columns=(headers + ["Accuracy", "AUC", "AP"]))


def store_metrics(metrics: dict, classes: list[str], fold, out_path: str):
    if len(classes) == 2:
        metric_df = _binary_metrics(metrics, classes)
    else:
        metric_df = _multiclass_metrics(metrics, classes)

    metric_df.index.name = f"Fold_{fold}"
    metric_df.to_csv(out_path, mode="a")
    return metric_df


def _binary_metrics(metrics: dict, classes: list[str]):
    return pd.DataFrame(
        {
            "PRECISION": np.hstack(
                (
                    metrics["precision_class"],
                    metrics["precision_macro"],
                    metrics["precision_weighted"],
                )
            ),
            "RECALL": np.hstack(
                (
                    metrics["recall_class"],
                    metrics["recall_macro"],
                    metrics["recall_weighted"],
                )
            ),
            "F1SCORE": np.hstack(
                (
                    metrics["fscore_class"],
                    metrics["fscore_macro"],
                    metrics["fscore_weighted"],
                )
            ),
            "ACCURACY": np.hstack(
                (
                    np.zeros(len(classes)),  # per-class = 0
                    metrics["accuracy"],
                    metrics["accuracy"],
                )
            ),
            "AUC": np.hstack(
                (
                    np.repeat(
                        metrics["auc_macro"], len(classes)
                    ),  # same for both classes
                    metrics["auc_macro"],
                    metrics["auc_weighted"],
                )
            ),
            "AP": np.hstack(
                (
                    np.repeat(metrics["ap_macro"], len(classes)),
                    metrics["ap_macro"],
                    metrics["ap_weighted"],
                )
            ),
        },
        index=classes + ["MACRO", "WEIGHTED"],
    )


def _multiclass_metrics(metrics: dict, classes: list[str]):
    return pd.DataFrame(
        {
            "PRECISION": np.hstack(
                (
                    metrics["precision_class"],
                    metrics["precision_macro"],
                    metrics["precision_weighted"],
                )
            ),
            "RECALL": np.hstack(
                (
                    metrics["recall_class"],
                    metrics["recall_macro"],
                    metrics["recall_weighted"],
                )
            ),
            "F1SCORE": np.hstack(
                (
                    metrics["fscore_class"],
                    metrics["fscore_macro"],
                    metrics["fscore_weighted"],
                )
            ),
            "ACCURACY": np.hstack(
                (np.zeros(len(classes)), metrics["accuracy"], metrics["accuracy"])
            ),
            "AUC": np.hstack(
                (metrics["auc_class"], metrics["auc_macro"], metrics["auc_weighted"])
            ),
            "AP": np.hstack(
                (metrics["ap_class"], metrics["ap_macro"], metrics["ap_weighted"])
            ),
        },
        index=classes + ["MACRO", "WEIGHTED"],
    )


# ------------------------ Evaluation ------------------------ #
def evaluate_multiclass_model(
    model,
    x_val,
    y_val,
    val_idx,
    fold,
    result_dir,
    data_model,
    classes,
    num_patients,
    time_opt,
):
    y_prob = model.predict_proba(x_val)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(
        y_val,
        y_pred,
        y_prob,
        len(classes),
    )

    metric_df = store_metrics(
        metrics,
        classes,
        fold,
        out_path=f"{result_dir}/metrics_{data_model}_{time_opt}_{num_patients}.csv",
    )

    # Save confusion matrix
    save_confusion_matrix(
        y_val,
        y_pred,
        f"{result_dir}/cm/cm_{data_model}_{time_opt}_{num_patients}_{fold}.jpg",
        labels=classes,
    )

    y_folder = f"{result_dir}/{fold}"
    os.makedirs(y_folder, exist_ok=True)
    np.save(f"{y_folder}/y_true.npy", y_val)
    np.save(f"{y_folder}/y_index.npy", val_idx)
    np.save(f"{y_folder}/y_pred.npy", y_pred)
    np.save(f"{y_folder}/y_prob.npy", y_prob)

    return metric_df

def run_tabulars_models(meds_root, outcomes_path, classes, result_dir, save_model = False):
    ROOT = meds_root

    df = build_features(pl.read_parquet(f"{ROOT}/data/**/0.parquet"))
    X = df.sort("subject_id").to_pandas().select_dtypes(exclude=["datetime64[ns]"])
    X = pd.get_dummies(X).drop(columns=["subject_id"])
    y = np.array(
        joblib.load(outcomes_path)
    )

    NUM_PATIENTS = len(y)
    CLASSES = classes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=len(classes),
            eval_metric="mlogloss",
            # feature_names=feature_names,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "lr": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        # multi_class="multinomial",
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        # "mlp": Pipeline(
        #     [
        #         ("imputer", SimpleImputer(strategy="median")),
        #         ("scaler", StandardScaler()),
        #         (
        #             "classifier",
        #             MLPClassifier(
        #                 hidden_layer_sizes=(256, 128),
        #                 activation="relu",
        #                 solver="adam",
        #                 alpha=1e-4,
        #                 batch_size=32,
        #                 learning_rate_init=1e-3,
        #                 max_iter=500,
        #                 early_stopping=True,
        #                 validation_fraction=0.1,
        #                 n_iter_no_change=20,
        #                 random_state=42,
        #                 verbose=False,
        #             ),
        #         ),
        #     ]
        # ),
    }

    best_score = -np.inf
    best_fold = None
    best_model = None

    for model_name, model in models.items():
        RESULTS = f"{result_dir}/{model_name}"

        os.makedirs(RESULTS, exist_ok=True)
        os.makedirs(f"{RESULTS}/cm", exist_ok=True)
        os.makedirs(f"{RESULTS}/models", exist_ok=True)

        all_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            x_train = X.iloc[train_idx]
            x_val = X.iloc[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            model.fit(x_train, y_train)

            metric = evaluate_multiclass_model(
                model,
                x_val,
                y_val,
                val_idx,
                fold,
                result_dir=RESULTS,
                data_model=model_name,
                classes=CLASSES,
                num_patients=NUM_PATIENTS,
                time_opt="TS",
            )

            all_metrics.append(metric)

            current_score = metric.loc["MACRO", "AUC"]

            if current_score > best_score: # type: ignore
                best_score = current_score
                best_fold = fold
                best_model = model

        if save_model:
            model_path = (
                f"{RESULTS}/models/"
                f"{model_name}_best_fold{best_fold}_auc_{best_score:.4f}.joblib"
            )

            joblib.dump(best_model, model_path)

            print(
                f"Saved best {model_name} model (fold={best_fold}, macro_auc={best_score:.4f})"
            )

        panel = pd.concat(all_metrics)
        metrics_mean = panel.groupby(level=0).mean()
        metrics_mean.index.name = "MEAN"
        metrics_std = panel.groupby(level=0).std()
        metrics_std.index.name = "STD"

        mean_std_metrics(metrics_mean, metrics_std, CLASSES).to_csv(
            f"{RESULTS}/metrics_TS_{NUM_PATIENTS}_mean_std.csv",
            sep="\t",
            index=False,
            mode="a",
        )
        metrics_mean.to_csv(f"{RESULTS}/metrics_TS_{NUM_PATIENTS}.csv", mode="a")
        metrics_std.to_csv(f"{RESULTS}/metrics_TS_{NUM_PATIENTS}.csv", mode="a")

    return X, y