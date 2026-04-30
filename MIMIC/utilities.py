import polars as pl
from pathlib import Path
import numpy as np
import os
import shutil
import pandas as pd

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


def generate_samples(
    n_folds: int,
    labels: pl.LazyFrame,
    size: int,
    seed: int | None,
    low_true_values=False,
):
    true_values = (
        labels.filter(pl.col("boolean_value"))
        .group_by("subject_id")
        .agg(pl.all().sample(n=1, seed=seed))
        .explode(pl.all().exclude("subject_id"))
        .collect(engine="streaming")
    )

    false_values = (
        labels.filter(~pl.col("boolean_value"))
        .join(
            true_values.select("subject_id").lazy(),
            on="subject_id",
            how="anti",
        )
        .group_by("subject_id")
        .agg(pl.all().sample(n=1, seed=seed))
        .explode(pl.all().exclude("subject_id"))
        .collect(engine="streaming")
    )

    fold_length = int(size / 2)

    false_values = false_values.sample(n=fold_length * n_folds, shuffle=True, seed=seed)
    false_samples = [
        false_values.slice(i * fold_length, fold_length) for i in range(n_folds)
    ]

    if low_true_values:
        true_samples = [
            true_values.sample(n=fold_length, shuffle=True, seed=seed)
            for _ in range(n_folds)
        ]
    else:
        true_values = true_values.sample(
            n=fold_length * n_folds, shuffle=True, seed=seed
        )

        true_samples = [
            true_values.slice(i * fold_length, fold_length) for i in range(n_folds)
        ]

    return [pl.concat([false_samples[i], true_samples[i]]) for i in range(n_folds)]


bad_strings = ["___", "None", "N/A", "", " "]

remove_long_text = (
    pl.when(
        (pl.col("text_value").is_in(bad_strings))
        | (pl.col("text_value").str.len_chars() > 50)
    )
    .then(None)
    .otherwise(pl.col("text_value"))
    .alias("text_value")
)

birthdate_to_age = [
    pl.when(pl.col("code") == "MEDS_BIRTH")
    .then((pl.col("prediction_time") - pl.col("time")) / pl.duration(days=365))
    .otherwise(pl.col("numeric_value"))  # <-- KEEP existing values
    .alias("numeric_value"),
    pl.when(pl.col("code") == "MEDS_BIRTH")
    .then(None)
    .otherwise(pl.col("time"))
    .alias("time"),
]

before_prediction_time = (pl.col("time").is_null()) | (
    pl.col("time") <= pl.col("prediction_time")
)

swap_text_with_numeric = {
    "numeric_value": pl.coalesce(pl.col("parsed"), pl.col("numeric_value")),
    "text_value": pl.when(pl.col("parsed").is_not_null())
    .then(None)
    .otherwise(pl.col("text_value")),
}

parse_txt_to_float = pl.col("text_value").cast(pl.Float64, strict=False)

regenerate_ids = (
    pl.col("subject_id")
    .cast(pl.Utf8)
    .cast(pl.Categorical)
    .to_physical()
    .alias("subject_id")
)

# no_vitals = ~pl.col("code").str.contains_any(vitals)

is_bp = pl.col("code") == "Blood Pressure"

# quick sanity check: contains a slash + strict-looking numeric pattern "num/num"
bp_text_valid = pl.col("text_value").is_not_null() & pl.col("text_value").str.contains(
    r"^\s*\d+(\.\d+)?/\d+(\.\d+)?\s*$"
)

# produce a list: for Blood Pressure valid texts -> [high, low] (floats, sorted desc)
# otherwise -> single-element list [numeric_value] so explode() yields one row for non-BP rows
bp_values_list = (
    pl.when(is_bp & bp_text_valid)
    .then(
        pl.col("text_value")
        .str.strip_chars()
        .str.split("/")
        .list.eval(
            pl.element().cast(pl.Float64, strict=False)
        )  # cast both parts to float
        .list.sort(descending=True)  # higher first
    )
    .otherwise(pl.concat_list([pl.col("numeric_value")]))
)


to_systolic_and_dyastolic_pressure = (
    pl.when(pl.col("orig_code").str.contains("Blood Pressure"))
    .then(
        pl.when(
            pl.col("numeric_value")
            == pl.col("bp_values").max().over(["subject_id", "time", "orig_text"])
        )
        .then(pl.lit("Systolic blood pressure"))
        .otherwise(pl.lit("Diastolic blood pressure"))
    )
    .otherwise(pl.col("code"))
)


set_BP_values_text_to_null = (
    pl.when(pl.col("orig_code").str.contains("Blood Pressure"))
    .then(pl.lit(None).cast(pl.Utf8))
    .otherwise(pl.col("text_value"))
)

derived_columns = ["bp_values", "orig_code", "orig_text", "parsed"]


def extract_labels_array(outcomes: pl.DataFrame):
    labels = (
        outcomes.select(["subject_id", "boolean_value"])
        .group_by("subject_id")
        .agg(pl.col("boolean_value").first().cast(pl.Int8).alias("boolean_value"))
        .sort("subject_id")
        .select("boolean_value")
        .to_series()
        .to_list()
    )
    return labels


window_dict = {
    "24h": pl.col("time").is_null()
    | (
        (pl.col("time") >= (pl.col("prediction_time") - pl.duration(days=1)))
        & (pl.col("time") <= pl.col("prediction_time"))
    ),
    "48h": pl.col("time").is_null()
    | (
        (pl.col("time") >= (pl.col("prediction_time") - pl.duration(days=2)))
        & (pl.col("time") <= pl.col("prediction_time"))
    ),
    "full": pl.col("time").is_null() | (pl.col("time") <= pl.col("prediction_time")),
}


def process_codes(input_parquet: str, output_dir: str, prefix_map: dict):
    """
    Extract codes for each ontology, strip prefix, flatten, and save efficiently.

    Args:
        input_parquet: path to the Parquet file containing 'parent_codes' column.
        output_dir: folder to save processed arrays.
        prefix_map: dict of {prefix_name: prefix_string}, e.g. "ICD10PCS": "ICD10PCS/"
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load Parquet lazily
    df = pl.scan_parquet(input_parquet).select(["parent_codes"])

    for name, prefix in prefix_map.items():
        print(f"Processing {name}...")

        # Lazy filter: keep lists that contain at least one code with prefix
        codes = (
            df.explode("parent_codes")
            .filter(pl.col("parent_codes").str.starts_with(prefix))
            .select(
                [
                    pl.col("parent_codes")
                    .str.replace(prefix, "", literal=True)
                    .alias("code")
                ]
            )
        )

        # Collect to memory and save as .npy
        codes_array = codes.collect()["code"].to_numpy()
        if len(codes_array) > 0:
            np.save(output_dir_path / f"{name}_codes.npy", codes_array)
            print(
                f"Saved {len(codes_array)} codes for {name} → {(name + '_codes.npy')}"
            )


PREFIX_MAP = {
    "ICD10PCS": "ICD10PCS/",
    "ICD10CM": "ICD10CM/",
    "LOINC": "LNC/",
    "RXNORM": "RXNORM/",
    "ICD9CM": "ICD9CM/",
    "SNOMED": "SNOMED/",
}


def freq_codes(events: pl.LazyFrame, quantile=0.15, inv=False):
    op = pl.col("len").lt if inv else pl.col("len").gt

    return (
        events.group_by("code")
        .len()
        .filter(op(pl.col("len").quantile(quantile)))
        .select("code")
    )


def freq_codes_v(events: pl.LazyFrame, freq=10):
    return events.group_by("code").len().filter(pl.col("len") > freq).select("code")


def aggregate_events(events: pl.LazyFrame, agg: str = "6h"):
    return (
        events.with_columns((pl.col("time").dt.truncate(agg)).alias("time"))
        .group_by(["subject_id", "code", "time"])
        .agg(
            [
                pl.col("numeric_value").mean().alias("numeric_value"),
                #pl.col("text_value").first().alias("text_value"),
                pl.col("prediction_time").first().alias("prediction_time"),
                pl.col("boolean_value").first().alias("boolean_value"),
            ]
        )
        .sort(["subject_id", "time"])
    )

def split_events(dt: pl.DataFrame):
    subjects = (
        dt.select("subject_id").unique().sample(fraction=1.0, shuffle=True, seed=1234)
    )

    n = subjects.height

    train_end = int(0.8 * n)
    held_out_end = int(0.9 * n)

    # Assign splits
    subjects = subjects.with_columns(
        pl.when(pl.arange(0, n) < train_end)
        .then(pl.lit("train"))
        .when(pl.arange(0, n) < held_out_end)
        .then(pl.lit("held_out"))
        .otherwise(pl.lit("tuning"))
        .alias("split")
    )

    events = dt.join(subjects, on="subject_id", how="left")

    labels = (
        events.select(["subject_id", "prediction_time", "boolean_value"])
        .group_by("subject_id")
        .agg(
            pl.col("boolean_value").first().cast(pl.Int8).alias("boolean_value"),
            pl.col("prediction_time").first().alias("prediction_time"),
        )
        .sort("subject_id")
        .join(subjects, on="subject_id", how="left")
    )
    return (subjects, events, labels)


def create_meds_cohort(
    events: pl.DataFrame,
    orig_dir: str,
    output_dir: str,
    columns: list[str] = ["subject_id", "code", "time", "numeric_value", "text_value"],
):
    (split_s, split_e, split_l) = split_events(events)

    split_s.write_parquet(f"{output_dir}/metadata/subject_splits.parquet")

    shutil.copy(
        f"{orig_dir}/metadata/codes.parquet", f"{output_dir}/metadata/codes.parquet"
    )
    shutil.copy(
        f"{orig_dir}/metadata/dataset.json", f"{output_dir}/metadata/dataset.json"
    )

    for split in ["train", "held_out", "tuning"]:
        df_events = split_e.filter(pl.col("split") == split)
        df_labels = split_l.filter(pl.col("split") == split)
        os.makedirs(f"{output_dir}/data/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
        df_events.select(columns).write_parquet(f"{output_dir}/data/{split}/0.parquet")
        df_labels.select(
            ["subject_id", "prediction_time", "boolean_value"]
        ).write_parquet(f"{output_dir}/labels/{split}/0.parquet")

    return (split_s, split_e, split_l)


def compute_metrics(y_true, y_pred, y_proba):
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_proba)

    # AUC can fail if only one class present
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan

    return {
        "f1_false": f1_per_class[0],  # type: ignore
        "f1_true": f1_per_class[1],  # type: ignore
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": accuracy,
        "auc": auc,
        "ap": ap,
    }


def aggregate_metrics(dict_metrics: dict):
    tasks_summary = {}

    for task, runs in dict_metrics.items():
        if not runs:
            print(f"⚠️ No valid runs for task: {task}")
            continue

        metrics_names = runs[0].keys()
        summary = {}

        for m in metrics_names:
            values = np.array([r[m] for r in runs], dtype=float)

            summary[m] = {
                "mean": round(np.nanmean(values), 2),
                "std": round(np.nanstd(values), 2),
            }

        tasks_summary[task] = summary

    return tasks_summary


def pretty_metrics_to_csv(metrics: dict, output_path: Path):
    rows = []

    for task, metrics in metrics.items():
        row = {"task": task}

        for m, stats in metrics.items():
            row[m] = f"{stats['mean']:.2f} ± {stats['std']:.2f}"

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False, sep="\t")


def get_latest_run(path: Path) -> Path | None:
    runs = [p for p in path.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def get_metrics_file_path(base_path: Path) -> Path | None:
    if not base_path.exists():
        print(f"⚠️ Missing path: {base_path}")
        return None

    latest_run = get_latest_run(base_path)

    if latest_run is None:
        print(f"⚠️ No runs found in: {base_path}")
        return None

    file_path = latest_run / "best_trial" / "held_out_predictions.parquet"

    if not file_path.exists():
        print(f"⚠️ Missing file: {file_path}")
        return None

    return file_path
