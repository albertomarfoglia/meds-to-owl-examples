import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

def _gen_start_event(y_min=2020, y_max=2023, rng=None):
    if rng is None: rng = np.random
    n_days = (y_max - y_min) * 365
    d0 = datetime.fromisoformat(f"{y_min}-01-01")
    delta = timedelta(
        days=int(rng.uniform(0, n_days)),
        hours=round(norm.rvs(12, 5)),
        minutes=round(np.random.uniform(60)),
    )
    return d0 + delta

# main preproc to generate times
def _create_event_times(df, event_cols, admission_col='admission', rng = np.random.RandomState(0)):
    df = df.copy()
    df[admission_col] = [_gen_start_event(rng=rng) for _ in range(len(df))]
    # 2) ensure sequence of event zeros become 1 (your rule)
    for ev in event_cols:
        time_col = f"{ev}_time"
        df[ev] = pd.to_numeric(df[ev], errors='coerce')
        # treat -1 as NaN -> no event
        df.loc[df[ev] < 0, ev] = np.nan
        # if value is exactly 0 -> set to 1
        df.loc[df[ev] == 0, ev] = 1
        # compute event datetime: admission + hours(sequence)
        df[ev] = df.apply(
            lambda r: (r[admission_col] + timedelta(hours=float(r[ev])))
            if pd.notna(r[ev]) and pd.notna(r[admission_col])
            else pd.NaT,
            axis=1
        )
    return df

def generate_meds_preprocessed(
    df : pd.DataFrame,
    output_path: str | None = None
) -> pd.DataFrame:
    df = df.copy()

    df['subject_id'] = df.index

    df["age"] = round(df["age"])
    df["hospital_stay_length"] = round(df["hospital_stay_length"])
    df["gcs"] = round(df["gcs"], 2)
    df["nb_acte"] = round(df["nb_acte"])

    event_cols = ['nimodipine','paracetamol','nad','corotrop','morphine','dve','atl','iot']
    df = _create_event_times(df, event_cols)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df