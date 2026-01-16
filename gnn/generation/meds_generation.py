from pathlib import Path
import numpy as np
import joblib
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
def _create_event_times(dff, event_cols, admission_col='admission', rng = np.random.RandomState(0)):
    #df = df.copy()
    df = dff[["subject_id"]].copy()
    df[admission_col] = [_gen_start_event(rng=rng) for _ in range(len(df))]
    # 2) ensure sequence of event zeros become 1 (your rule)
    for ev in event_cols:
        time_col = f"{ev}_time"
        df[ev] = pd.to_numeric(dff[ev], errors='coerce')
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
        df.dropna(subset=[ev]) # remove nan values
    return df

class EventsTable():
    def __init__(self, data: pd.DataFrame):
        self.proc_codes = {
            "dve": "00P6X0Z",
            "atl": "Z98.6",
            "iot": "0BH17EZ"
        }

        self.admin_codes = {
            "nimodipine": "C08CA06",
            "paracetamol": "N02BE01",
            "nad": "C01CA03",
            "corotrop": "C01CE02",
            "morphine": "N02AA01",
        }

        self.names = np.array(list(self.proc_codes.keys()) + list(self.admin_codes.keys()))

        codes = self.proc_codes | self.admin_codes

        events = _create_event_times(data, event_cols=codes.keys()).to_dict()

        adms = pd.DataFrame(events).melt(
            id_vars=["subject_id", "admission"],
            var_name="name",
            value_name="time"
        )

        adms["code"] = adms["name"].map(lambda x: codes[x])

        self.df = adms.dropna(subset=["time"]).reset_index(drop=True)
             
    def to_administrations(self):
        return self.df[self.df["code"].isin(self.admin_codes.values())]

    def to_procedures(self):
        return self.df[self.df["code"].isin(self.proc_codes.values())]

def generate_meds_preprocessed(
    df : pd.DataFrame,
    output_path: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df['subject_id'] = df.index

    df["age"] = round(df["age"])
    df["hospital_stay_length"] = round(df["hospital_stay_length"])
    df["gcs"] = round(df["gcs"], 2)
    df["nb_acte"] = round(df["nb_acte"])

    events = EventsTable(df)
    df_admin = events.to_administrations()
    df_proc = events.to_procedures()
    df_pat = df.drop(columns=events.names)

    print(df_pat.columns)

    if output_path:
        df_pat.to_parquet(f"{output_path}/patients.parquet", index=False)
        df_admin.to_parquet(f"{output_path}/administrations.parquet", index=False)
        df_proc.to_parquet(f"{output_path}/procedures.parquet", index=False)

    return (df_pat, df_admin, df_proc)

from MEDS_transforms.runner import main
from meds2rdf import MedsRDFConverter
import shutil
import os

MEDS_ETL_OUTPUT = "meds_output"

def gen_meds_kg(num_patients: int, data_path: Path, time_opt="TS"):
    df = pd.read_csv(data_path, index_col=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    no_outcome = df.drop(columns=["outcome"])

    os.makedirs(MEDS_ETL_OUTPUT, exist_ok=True)
    os.makedirs("intermediate", exist_ok=True)
    generate_meds_preprocessed(no_outcome, output_path="intermediate")

    shutil.rmtree(MEDS_ETL_OUTPUT)

    main([
        "pkg://MEDS_extract.configs._extract.yaml",
        "--overrides",
        "input_dir=intermediate",
        "output_dir=output",
        "event_conversion_config_fp=../MESSY.yaml",
        "dataset.name=Neurovasc",
        "dataset.version=1.0",
    ])


    # Initialize the converter with the path to your MEDS dataset directory
    converter = MedsRDFConverter("output")
    graph = converter.convert(include_dataset_metadata=False)

    graph.serialize(destination=f"data/meds_{time_opt}_{num_patients}.nt", format="nt")

    joblib.dump(df["outcome"].astype(int).to_list(), f"data/outcomes_meds_{time_opt}_{num_patients}.joblib")
