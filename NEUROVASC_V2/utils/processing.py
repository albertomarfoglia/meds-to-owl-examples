import polars as pl
import shutil
import os

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
    return (subjects, events)

def create_meds_cohort(
    events: pl.DataFrame,
    orig_dir: str,
    output_dir: str,
    columns: list[str] = ["subject_id", "code", "time", "numeric_value", "text_value"],
):
    split_s, split_e = split_events(events)

    split_s.write_parquet(f"{output_dir}/metadata/subject_splits.parquet")

    filtered_codes = events.select("code").unique()
    pl.read_parquet(f"{orig_dir}/metadata/codes.parquet").join(
        filtered_codes, on="code", how="inner"
    ).write_parquet(f"{output_dir}/metadata/codes.parquet")

    shutil.copy(
        f"{orig_dir}/metadata/dataset.json", f"{output_dir}/metadata/dataset.json"
    )

    for split in ["train", "held_out", "tuning"]:
        df_events = split_e.filter(pl.col("split") == split)
        os.makedirs(f"{output_dir}/data/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
        df_events.select(columns).write_parquet(f"{output_dir}/data/{split}/0.parquet")

    return (split_s, split_e)