import pandas as pd
import joblib
from .neurovasc_meta import CONTEXTUAL_VARIABLES, SEQUENTIAL_VARIABLES, KEY_VARIABLES
import numpy as np

def generate_meds_preprocessed(
    df : pd.DataFrame,
    output_path: str | None = None,
    outcome_path: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _df = df.copy()

    pat_to_id = {k: v for v, k in enumerate(set(_df["Patient_ID"]), start=0)}
    _df["Patient_ID"] = _df["Patient_ID"].map(pat_to_id)
    _df["INDEX"] = _df["Patient_ID"]
    _df = _df.set_index("INDEX")
    
    _df["Timestamp"] = pd.to_datetime(_df["Timestamp"], errors="coerce")
    _df[SEQUENTIAL_VARIABLES] = _df[SEQUENTIAL_VARIABLES].replace(False, np.nan)

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

    return (df_contextual, df_sequential, df_outcomes)