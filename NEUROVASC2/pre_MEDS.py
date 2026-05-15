import pandas as pd
import joblib
from .neurovasc_meta import CONTEXTUAL_VARIABLES, SEQUENTIAL_VARIABLES, KEY_VARIABLES
import polars as pl

def generate_meds_preprocessed(
    df : pl.DataFrame,
    output_path: str | None = None,
    outcome_path: str | None = None,
    synthetic = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _df = df.to_pandas().copy()

    # TODO To remove at the end
    if synthetic:
        #_df.drop(columns=[""], inplace=True)
        CVARIABLES = [
            x for x in CONTEXTUAL_VARIABLES
            if x not in {"Length_of_Stay", "Number_of_Visited_Departments"}
        ]
    else:
        CVARIABLES = CONTEXTUAL_VARIABLES

    pat_to_id = {k: v for v, k in enumerate(set(_df["Patient_ID"]), start=0)}
    _df["Patient_ID"] = _df["Patient_ID"].map(pat_to_id)
    _df["INDEX"] = _df["Patient_ID"]
    _df = _df.set_index("INDEX")
    
    _df["Timestamp"] = pd.to_datetime(_df["Timestamp"], errors="coerce")
    #_df[SEQUENTIAL_VARIABLES] = _df[SEQUENTIAL_VARIABLES].replace(False, np.nan)

    df_patients = _df[KEY_VARIABLES + CVARIABLES].drop_duplicates(subset='Patient_ID', keep='first')
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