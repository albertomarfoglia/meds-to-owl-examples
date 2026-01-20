import pandas as pd
from collections import defaultdict
from rdflib import Graph
import pandas as pd
from .queries import run_query, PATIENT_FEATURE_MATRIX
from .neurovasc_meta import *

def get_matching_prefix(code, prefixes):
    return next((p for p in prefixes if code.startswith(p)), None)

def build_neurovasc_meds_dt(records):
    data = defaultdict(dict)

    for r in records: # initial event values to 0
        sid = r["subject_id"]
        data[sid].update({col: 0 for col in EVENTS_COLUMNS})

    for r in records:
        sid = r["subject_id"]
        code = r["code"]

        if (k := get_matching_prefix(code, NUMERIC_CODES)):
            data[sid][NUMERIC_CODES[k]] = r["numeric_value"]

        elif (k := get_matching_prefix(code, CATEGORICAL_PREFIXES)):
            data[sid][CATEGORICAL_PREFIXES[k]] = int(code.split("//")[1])

        elif (k := get_matching_prefix(code, BINARY_PREFIXES)):
            data[sid][BINARY_PREFIXES[k]] = int(code.split("//")[1])

        elif icd_codes.get(code) is not None:
             data[sid][icd_codes.get(code)] = 1

        elif atc_codes.get(code) is not None:
             data[sid][atc_codes.get(code)] = 1
        
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(columns=FEATURE_COLUMNS)
    df.index.name = "sub_id"

    return df.sort_index()

def build_neurovasc_medskg_dt(graph: Graph):
    rows = run_query(graph, "all features", PATIENT_FEATURE_MATRIX)
    df = pd.DataFrame(rows)
    df["sub_id"] = df["sub_id"].astype(int)
    return df.set_index("sub_id").sort_index()

def check_dts_consistency(dt1: pd.DataFrame, dt2: pd.DataFrame):
    common_index = dt1.index.intersection(dt2.index)
    common_cols = dt1.columns.intersection(dt2.columns)

    dfm1 = dt1.loc[common_index, common_cols].sort_index()
    dfs1 = dt2.loc[common_index, common_cols].sort_index()

    diff = dfm1.compare(dfs1)
    print(f"Total features: {len(common_cols)} checked -> diff: [{len(diff)}]")
    return diff