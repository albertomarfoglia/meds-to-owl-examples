import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

from .neurovasc_meta import EVENTS_COLUMNS, icd_codes, atc_codes

def _gen_start_event(y_min=2020, y_max=2023):
    n_days = (y_max - y_min) * 365
    d0 = datetime.fromisoformat(f"{y_min}-01-01")
    day_rand = round(np.random.uniform(n_days))
    delta = timedelta(
        days=day_rand,
        hours=round(norm.rvs(12, 5)),
        minutes=round(np.random.uniform(60)),
    )
    return d0 + delta

def _create_event_times(df: pd.DataFrame) -> pd.DataFrame: 
    _df = df[["subject_id"]].copy()
    _df["admission"] = [_gen_start_event() for _ in range(len(_df))]
    for ev in EVENTS_COLUMNS:
        offsets = pd.to_numeric(df[ev], errors="coerce")
        offsets.loc[offsets == 0] = 1

        _df[ev] = pd.NaT
        _df[ev] = pd.to_datetime(_df[ev])

        mask = offsets > 0
        _df.loc[mask, ev] = _df.loc[mask, "admission"] + pd.to_timedelta(offsets.loc[mask], unit="h")

    _df.drop(columns=["admission"], inplace=True)
    return _df

class Preprocessor():
    def __init__(self, data: pd.DataFrame):
        _df = data.copy()
        _df['subject_id'] = data.index
        _df["hospital_stay_length"] = data["hospital_stay_length"].round()
        _df["nb_acte"] = data["nb_acte"].round()
        _df["age"] = data["age"].round()
        _df["gcs"] = data["gcs"].round(2) 

        self.icd_codes = {
            v: k.replace("ICD//10//", "")
            for k, v in icd_codes.items()
        }

        self.atc_codes = {
            v: k.replace("ATC//", "")
            for k, v in atc_codes.items()
        }

        events = _create_event_times(_df)

        _df_events: pd.DataFrame = (
            events
            .melt(
                id_vars=["subject_id"],
                value_vars=EVENTS_COLUMNS,
                var_name="name",
                value_name="time",
            )
            .dropna(subset=["time"])
            .sort_values(["subject_id"])
            .reset_index(drop=True)
        )

        _df_events["code"] = _df_events["name"].map(lambda x: (self.icd_codes | self.atc_codes)[x])

        self.df = _df
        self.df_events = _df_events
             
    def to_administrations(self):
        return self.df_events[self.df_events["code"].isin(self.atc_codes.values())]

    def to_procedures(self):
        return self.df_events[self.df_events["code"].isin(self.icd_codes.values())]
    
    def to_patients(self):
        return self.df.drop(columns=EVENTS_COLUMNS)

def generate_meds_preprocessed(
    df : pd.DataFrame,
    output_path: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    events = Preprocessor(df)
    df_admin = events.to_administrations()
    df_proc = events.to_procedures()
    df_pat = events.to_patients()

    if output_path:
        df_pat.to_parquet(f"{output_path}/patients.parquet", index=False)
        df_admin.to_parquet(f"{output_path}/administrations.parquet", index=False)
        df_proc.to_parquet(f"{output_path}/procedures.parquet", index=False)

    return (df_pat, df_admin, df_proc)