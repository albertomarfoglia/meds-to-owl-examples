import pandas as pd
import joblib
from .neurovasc_meta import CONTEXTUAL_VARIABLES, SEQUENTIAL_VARIABLES, KEY_VARIABLES

def generate_meds_preprocessed(
    df : pd.DataFrame,
    output_path: str | None = None,
    outcome_path: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _df = df.copy()

    pat_to_id = {k: v for v, k in enumerate(set(_df["ID_PAT"]), start=0)}
    _df["ID_PAT"] = _df["ID_PAT"].map(pat_to_id)
    _df["INDEX"] = _df["ID_PAT"]
    _df = _df.set_index("INDEX")
    
    _df["DATE"] = pd.to_datetime(_df["DATE"], errors="coerce")
    _df[SEQUENTIAL_VARIABLES] = _df[SEQUENTIAL_VARIABLES].replace(False, pd.NA)

    df_patients = _df[KEY_VARIABLES + CONTEXTUAL_VARIABLES].drop_duplicates(subset='ID_PAT', keep='first')
    df_patients = df_patients.sort_index()

    df_contextual = df_patients.drop(columns=["outcomes"])
    df_sequential = _df[KEY_VARIABLES + SEQUENTIAL_VARIABLES].sort_index()
    df_outcomes = df_patients["outcomes"].astype(int)

    if output_path:
        df_contextual.to_parquet(f"{output_path}/contextual.parquet", index=False)
        df_sequential.to_parquet(f"{output_path}/sequential.parquet", index=False)
    if outcome_path:
        joblib.dump(df_outcomes.to_list(), outcome_path)

    return (df_contextual, df_sequential, pd.DataFrame(df_outcomes))