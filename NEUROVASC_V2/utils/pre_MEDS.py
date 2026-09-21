import pandas as pd
import joblib
from .neurovasc_meta import CONTEXTUAL_VARIABLES, SEQUENTIAL_VARIABLES, KEY_VARIABLES
import polars as pl
import numpy as np
from datetime import datetime, timedelta

def generate_meds_preprocessed(
    df : pl.DataFrame,
    output_path: str | None = None,
    outcome_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _df = df.to_pandas().copy()

    pat_to_id = {k: v for v, k in enumerate(set(_df["Patient_ID"]), start=0)}
    _df["Patient_ID"] = _df["Patient_ID"].map(pat_to_id)
    _df["INDEX"] = _df["Patient_ID"]
    _df = _df.set_index("INDEX")
    
    _df["Timestamp"] = pd.to_datetime(_df["Timestamp"], errors="coerce")
    #_df[SEQUENTIAL_VARIABLES] = _df[SEQUENTIAL_VARIABLES].replace(False, np.nan)

    df_patients = _df[KEY_VARIABLES + CONTEXTUAL_VARIABLES].drop_duplicates(subset='Patient_ID', keep='first')
    df_patients = df_patients.sort_index()

    df_contextual = df_patients.drop(columns=["Outcome"])
    df_sequential = _df[KEY_VARIABLES + SEQUENTIAL_VARIABLES].sort_index()

    outcome_mapping = {
        "DOMICILE": 0,
        "REEDUC_TRANSFERT": 1,
        "DECES": 2,
    }

    df_outcomes = df_patients["Outcome"].map(outcome_mapping).astype(int)

    if output_path:
        df_contextual.to_parquet(f"{output_path}/contextual.parquet", index=False)
        df_sequential.to_parquet(f"{output_path}/sequential.parquet", index=False)
    if outcome_path:
        joblib.dump(df_outcomes.to_list(), outcome_path)

    return (df_contextual, df_sequential, df_outcomes) # type: ignore



def rebalance_synth(df_input: pl.DataFrame, n_patients = 5000):
    # ----------------------------
    # Step 1: Build patient-level table
    # (assumes each patient has a single Outcome)
    # ----------------------------
    patient_df = (
        df_input
        .group_by("Patient_ID")
        .agg(
            pl.col("Outcome").first().alias("Outcome")
        )
    )

    # ----------------------------
    # Step 2: Define target distribution
    # ----------------------------
    target_n = n_patients

    target_dist = {
        "DECES": 0.141,
        "REEDUC_TRANSFERT": 0.386,
        "DOMICILE": 0.473,
    }

    target_counts = {
        k: int(v * target_n)
        for k, v in target_dist.items()
    }

    # fix rounding error
    diff = target_n - sum(target_counts.values())
    target_counts["DOMICILE"] += diff

    # ----------------------------
    # Step 3: Check availability (important safeguard)
    # ----------------------------
    available = (
        patient_df
        .group_by("Outcome")
        .len()
    )

    available_dict = dict(zip(available["Outcome"], available["len"]))

    for k, v in target_counts.items():
        if v > available_dict.get(k, 0):
            raise ValueError(
                f"Not enough patients in class {k}: "
                f"requested {v}, available {available_dict.get(k, 0)}"
            )

    # ----------------------------
    # Step 4: Stratified sampling at patient level
    # ----------------------------
    sampled_patients = pl.concat([
        patient_df.filter(pl.col("Outcome") == outcome)
        .sample(n=n, seed=42)
        for outcome, n in target_counts.items()
    ])

    # ----------------------------
    # Step 5: Map back to full dataset
    # ----------------------------
    df_sampled = (
        df_input
        .join(sampled_patients.select("Patient_ID"),
            on="Patient_ID",
            how="inner")
    )

    # ----------------------------
    # Optional sanity check
    # ----------------------------
    print(
        df_sampled
        .group_by("Patient_ID")
        .first()
        .group_by("Outcome")
        .len()
        .sort("Outcome")
    )

    print("Number of patients:", sampled_patients.height)
    print("Number of rows:", df_sampled.height)

    return df_sampled

def generate_patient_timestamps(
    df: pl.DataFrame,
    id_col: str = "Patient_ID",
    time_col: str = "Relative_Time",
    output_col: str = "Timestamp",
    start_min: datetime = datetime(2020, 1, 1),
    start_max: datetime = datetime(2023, 1, 1),
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Assigns a random base timestamp per patient and computes absolute timestamps.
    """

    if seed is not None:
        np.random.seed(seed)

    delta_days = (start_max - start_min).days

    def random_date():
        return start_min + timedelta(days=np.random.randint(delta_days))

    base_dates = (
        df.select(id_col)
        .unique()
        .with_columns(
            pl.col(id_col)
            .map_elements(lambda _: random_date())
            .alias("base_ts")
        )
    )

    return (
        df.join(base_dates, on=id_col)
        .with_columns(
            (
                pl.col("base_ts") + pl.duration(days=pl.col(time_col))
            ).alias(output_col)
        )
        .drop("base_ts")
    )