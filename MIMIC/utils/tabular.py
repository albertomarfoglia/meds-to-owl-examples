import polars as pl
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from codecarbon import EmissionsTracker
import time

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

# --- import (or copy) the exact same split function the RGCN pipeline uses ---
def k_fold(X, y, folds, random_state=77):
    skf = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    train_indices, val_indices, test_indices = [], [], []
    train_y, val_y, test_y = [], [], []
    for non_test_idx, test_idx in skf.split(X, y):
        test_indices.append(X[test_idx])
        train_idx, val_idx, _, _ = train_test_split(
            non_test_idx, y[non_test_idx], test_size=1 / 9, random_state=random_state
        )
        train_indices.append(X[train_idx])
        val_indices.append(X[val_idx])
        train_y.append(y[train_idx])
        val_y.append(y[val_idx])
        test_y.append(y[test_idx])
    return train_indices, val_indices, test_indices, train_y, val_y, test_y


def run_tabulars_models(meds_root, outcomes_path, classes, result_dir, save_model=False):
    ROOT = meds_root

    df = build_features(pl.read_parquet(f"{ROOT}/data/**/0.parquet"))
    X_df = df.sort("subject_id").to_pandas().select_dtypes(exclude=["datetime64[ns]"])
    X_df = pd.get_dummies(X_df).drop(columns=["subject_id"])
    X = X_df.to_numpy()
    y = np.array(joblib.load(outcomes_path))

    NUM_PATIENTS = len(y)
    CLASSES = classes

    # --- CHANGED: same 10-fold, same random_state, same train/val/test split
    # the RGCN pipeline uses, instead of an independent 5-fold split. ---
    train_idx_list, val_idx_list, test_idx_list, train_y_list, val_y_list, test_y_list = k_fold(
        np.arange(len(y)), y, folds=10, random_state=77
    )

    models_config = {
        "xgboost": lambda: XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="multi:softprob" if len(classes) > 2 else "binary:logistic",
            num_class=len(classes) if len(classes) > 2 else None,
            eval_metric="mlogloss" if len(classes) > 2 else "logloss",
            early_stopping_rounds=20,  # CHANGED: mirrors RGCN's val-based early stopping
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=10, random_state=42, n_jobs=-1,
        ),
        "lr": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="lbfgs", max_iter=5000, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }

    best_score = {name: -np.inf for name in models_config}
    best_fold = {name: None for name in models_config}
    best_model = {name: None for name in models_config}

    for model_name, model_factory in models_config.items():
        RESULTS = f"{result_dir}/{model_name}"
        os.makedirs(RESULTS, exist_ok=True)
        os.makedirs(f"{RESULTS}/cm", exist_ok=True)
        os.makedirs(f"{RESULTS}/models", exist_ok=True)

        all_metrics = []

        for fold in range(10):
            tracker = EmissionsTracker(
                project_name=f"fold_{fold}",
                output_dir=result_dir,
                measure_power_secs=1,
                log_level="critical",
            )

            tracker.start()
            start = time.perf_counter()

            train_idx = train_idx_list[fold]
            val_idx = val_idx_list[fold]
            test_idx = test_idx_list[fold]  # CHANGED: real held-out test fold, distinct from val

            x_train, y_train = X[train_idx], y[train_idx]
            x_val, y_val = X[val_idx], y[val_idx]
            x_test, y_test = X[test_idx], y[test_idx]

            model = model_factory()

            # --- CHANGED: XGBoost gets the same validation-based stopping signal
            # the RGCN receives; RF/LR have no native equivalent (see note below). ---
            if model_name == "xgboost":
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            else:
                model.fit(x_train, y_train)

            # CHANGED: evaluate on the *test* fold, not the validation fold
            metric = evaluate_multiclass_model(
                model, x_test, y_test, test_idx, fold,
                result_dir=RESULTS, data_model=model_name,
                classes=CLASSES, num_patients=NUM_PATIENTS, time_opt="TS",
            )
            all_metrics.append(metric)

            current_score = metric.loc["MACRO", "AUC"]
            if current_score > best_score[model_name]: # type: ignore
                best_score[model_name] = current_score  # type: ignore
                best_fold[model_name] = fold  # type: ignore
                best_model[model_name] = model  # type: ignore

            runtime = time.perf_counter() - start
            emissions = tracker.stop()

            print(f"Runtime: {runtime}", f"Emissions: {emissions}")

        if save_model:
            model_path = (
                f"{RESULTS}/models/"
                f"{model_name}_best_fold{best_fold[model_name]}_auc_{best_score[model_name]:.4f}.joblib"
            )
            joblib.dump(best_model[model_name], model_path)

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

        print(mean_std_metrics(metrics_mean, metrics_std, CLASSES))

    return X_df, y


# def extract_float(text):
#     """Extract first float from messy string"""
#     if isinstance(text, (float, int)):
#         return float(text)

#     match = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
#     return float(match.group()) if match else np.nan


# def parse_mean_block(file_path):
#     df = pd.read_csv(file_path)

#     # keep only MEAN rows
#     df = df[df["MEAN"].isin(["FALSE", "TRUE", "MACRO", "WEIGHTED"])].copy()

#     df = df.set_index("MEAN")

#     clean = {}

#     for metric in ["F1SCORE", "ACCURACY", "AUC", "AP"]:
#         clean[metric] = {
#             row: extract_float(df.loc[row, metric])
#             for row in df.index
#         }

#     return clean

# def format_row(summary):
#     order = [
#         "FALSE_F1",
#         "TRUE_F1",
#         "MACRO_F1",
#         "WEIGHTED_F1",
#         "ACCURACY",
#         "AUC",
#         "AP"
#     ]

#     values = []
#     for k in order:
#         mean, std = summary[k]
#         values.append(f"{mean:.2f} ± {std:.2f}")

#     print("\t".join(order))
#     print("\t".join(values))


# def aggregate_task(exp, results_dir, model_type):
#     per_subsample = []

#     for s in range(0, exp["num_of_samples"]):
#         file_path = f"{results_dir}/{exp['task']}/meds/{str(s)}/metrics_{exp['sample_size']}/{model_type}/metrics_TS_{exp['sample_size']}.csv"

#         mean_block = parse_mean_block(file_path)

#         per_subsample.append({
#             "FALSE_F1": mean_block["F1SCORE"]["FALSE"],
#             "TRUE_F1": mean_block["F1SCORE"]["TRUE"],
#             "MACRO_F1": mean_block["F1SCORE"]["MACRO"],
#             "WEIGHTED_F1": mean_block["F1SCORE"]["WEIGHTED"],
#             "ACCURACY": mean_block["ACCURACY"]["MACRO"],
#             "AUC": mean_block["AUC"]["MACRO"],
#             "AP": mean_block["AP"]["MACRO"],
#         })

#     df = pd.DataFrame(per_subsample)

#     summary = {}
#     for col in df.columns:
#         values = df[col].values
#         summary[col] = (np.mean(values), np.std(values, ddof=1)) # type: ignore

#     return summary

def extract_float(text):
    """Extract first float from messy string."""
    if isinstance(text, (float, int)):
        return float(text)

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    return float(match.group()) if match else np.nan


def parse_mean_block(file_path):
    df = pd.read_csv(file_path)

    # Keep only MEAN rows
    df = df[df["MEAN"].isin(["FALSE", "TRUE", "MACRO", "WEIGHTED"])].copy()
    df = df.set_index("MEAN")

    clean = {}

    for metric in ["F1SCORE", "ACCURACY", "AUC", "AP"]:
        clean[metric] = {
            row: extract_float(df.loc[row, metric])
            for row in df.index
        }

    return clean


def parse_emissions(emissions_path):
    """
    Read emissions.csv for one subsample.

    The file contains 10 rows (fold_0 ... fold_9).
    We compute the mean duration and mean CO2 emissions
    across the 10 folds.
    """
    df = pd.read_csv(emissions_path)

    required_columns = {"duration", "emissions"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns {missing} in {emissions_path}"
        )

    # Convert to numeric in case the CSV contains strings
    duration = pd.to_numeric(df["duration"], errors="coerce")
    emissions = pd.to_numeric(df["emissions"], errors="coerce")

    return {
        "DURATION": duration.mean(),
        "CO2": emissions.mean(),
    }


def format_row(summary):
    order = [
        "FALSE_F1",
        "TRUE_F1",
        "MACRO_F1",
        "WEIGHTED_F1",
        "ACCURACY",
        "AUC",
        "AP",
        "DURATION",
        "CO2",
    ]

    values = []

    for k in order:
        mean, std = summary[k]

        # Metrics
        if k not in ["DURATION", "CO2"]:
            values.append(f"{mean:.2f} ± {std:.2f}")

        # Duration in seconds
        elif k == "DURATION":
            values.append(f"{mean:.2f} ± {std:.2f} s")

        # CO2 emissions in kg
        elif k == "CO2":
            values.append(f"{mean:.6f} ± {std:.6f} kg")

    print("\t".join(order))
    print("\t".join(values))


def aggregate_task(exp, results_dir, model_type):
    per_subsample = []

    for s in range(0, exp["num_of_samples"]):

        base_path = (
            results_dir
            / exp["task"]
            / "meds"
            / str(s)
            / f"metrics_{exp['sample_size']}"
        )

        metrics_path = (
            base_path
            / model_type
            / f"metrics_TS_{exp['sample_size']}.csv"
        )

        emissions_path = base_path / "emissions.csv"

        # -------------------------
        # Classification metrics
        # -------------------------
        mean_block = parse_mean_block(metrics_path)

        # -------------------------
        # Duration + CO2
        # -------------------------
        emissions_stats = parse_emissions(emissions_path)

        per_subsample.append({
            "FALSE_F1": mean_block["F1SCORE"]["FALSE"],
            "TRUE_F1": mean_block["F1SCORE"]["TRUE"],
            "MACRO_F1": mean_block["F1SCORE"]["MACRO"],
            "WEIGHTED_F1": mean_block["F1SCORE"]["WEIGHTED"],
            "ACCURACY": mean_block["ACCURACY"]["MACRO"],
            "AUC": mean_block["AUC"]["MACRO"],
            "AP": mean_block["AP"]["MACRO"],

            # Mean over the 10 folds of this subsample
            "DURATION": emissions_stats["DURATION"],
            "CO2": emissions_stats["CO2"],
        })

    df = pd.DataFrame(per_subsample)

    # -------------------------
    # Aggregate across subsamples
    # -------------------------
    summary = {}

    for col in df.columns:
        values = df[col].values

        # Mean across subsamples
        mean = np.mean(values) # type: ignore

        # Sample std across subsamples
        std = np.std(values, ddof=1) # type: ignore

        summary[col] = (mean, std)

    return summary