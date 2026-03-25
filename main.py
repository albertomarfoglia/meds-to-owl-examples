# import os
# import polars as pl

# import joblib

# from meds2rdf import MedsRDFConverter
# from meds2rdf.sinks import NTriplesSink
# from meds2rdf.config import Config, MEDSSchema
# from pathlib import Path

# MIMIC_ETL_OUTPUT = "MIMIC/MEDS_cohort"
# MIMIC_ETL_GRAPH = f"{MIMIC_ETL_OUTPUT}/graph"
# MIMIC_TASKS_PATH = "MIMIC/tasks"
# TIME_OPT = "TS"

# TMP_DIR = f"{MIMIC_ETL_OUTPUT}/tmp"
# TMP_DATA_DIR = f"{TMP_DIR}/data/train"
# TMP_METADATA_DIR = f"{TMP_DIR}/metadata"

# os.makedirs(TMP_DATA_DIR, exist_ok=True)
# os.makedirs(TMP_METADATA_DIR, exist_ok=True)
# os.makedirs(MIMIC_ETL_GRAPH, exist_ok=True)

# # long_term_reccurrence
# # inhospital_mortality
# # imminent_mortality
# TASK = "inhospital_mortality"

# IHM = f"{MIMIC_ETL_OUTPUT}/labels/{TASK}/**/*.parquet"
# EXPORT_DIR = f"exports/{TASK}"
# LABELS_DIR = f"{EXPORT_DIR}/labels"
# os.makedirs(EXPORT_DIR, exist_ok=True)
# os.makedirs(LABELS_DIR, exist_ok=True)


# outcomes = pl.scan_parquet(IHM).select("subject_id", "prediction_time", "boolean_value")

# true_values = (
#     outcomes.filter(pl.col("boolean_value"))
#     .sort(pl.col("prediction_time"), descending=True)
#     .group_by("subject_id")
#     .head(1)
#     .collect(engine="streaming")
# )

# false_values = (
#     outcomes.filter(~pl.col("boolean_value"))
#     .join(
#         true_values.select("subject_id").lazy(),
#         on="subject_id",
#         how="anti",
#     )
#     .group_by("subject_id")
#     .head(1)
#     .collect(engine="streaming")
# )

# # folds = 1

# # n_patients = 1800

# # true_values = true_values.sample(n=int(n_patients / 2), shuffle=True, seed=1234)
# # false_values = false_values.sample(
# #     n=int(n_patients / 2 * folds), shuffle=True, seed=1234
# # )

# # samples = [
# #     false_values.slice(i * len(true_values), len(true_values)).extend(true_values)
# #     for i in range(folds)
# # ]

# folds = 5

# n_patients = 1800  # len(true_values) * 2
# fold_length = int(n_patients / 2)

# true_values = true_values.sample(n=fold_length * folds, shuffle=True, seed=1234)
# false_values = false_values.sample(n=fold_length * folds, shuffle=True, seed=1234)

# true_samples = [true_values.slice(i * fold_length, fold_length) for i in range(folds)]

# false_samples = [false_values.slice(i * fold_length, fold_length) for i in range(folds)]

# samples = [false_samples[i].extend(true_samples[i]) for i in range(folds)]

# bad_strings = ["___", "None", "N/A", "", " "]

# remove_long_text = (
#     pl.when(
#         (pl.col("text_value").is_in(bad_strings))
#         | (pl.col("text_value").str.len_chars() > 50)
#     )
#     .then(None)
#     .otherwise(pl.col("text_value"))
#     .alias("text_value")
# )

# before_prediction_time = (pl.col("time").is_null()) | (
#     pl.col("time") <= pl.col("prediction_time")
# )

# swap_text_with_numeric = (
#     pl.coalesce(pl.col("parsed"), pl.col("numeric_value")),
#     pl.when(pl.col("parsed").is_not_null()).then(None).otherwise(pl.col("text_value")),
# )

# parse_txt_to_float = pl.col("text_value").cast(pl.Float64, strict=False)

# regenerate_ids = (
#     pl.col("subject_id")
#     .cast(pl.Utf8)
#     .cast(pl.Categorical)
#     .to_physical()
#     .alias("subject_id")
# )

# # no_vitals = ~pl.col("code").str.contains_any(vitals)

# is_bp = pl.col("code") == "Blood Pressure"

# # quick sanity check: contains a slash + strict-looking numeric pattern "num/num"
# bp_text_valid = pl.col("text_value").is_not_null() & pl.col("text_value").str.contains(
#     r"^\s*\d+(\.\d+)?/\d+(\.\d+)?\s*$"
# )

# # produce a list: for Blood Pressure valid texts -> [high, low] (floats, sorted desc)
# # otherwise -> single-element list [numeric_value] so explode() yields one row for non-BP rows
# bp_values_list = (
#     pl.when(is_bp & bp_text_valid)
#     .then(
#         pl.col("text_value")
#         .str.strip_chars()
#         .str.split("/")
#         .list.eval(
#             pl.element().cast(pl.Float64, strict=False)
#         )  # cast both parts to float
#         .list.sort(descending=True)  # higher first
#     )
#     .otherwise(pl.concat_list([pl.col("numeric_value")]))
# )




# for idx, sample in enumerate(samples):

#     n_patients = len(samples[0])

#     events_base = (
#         pl.scan_parquet(f"{MIMIC_ETL_OUTPUT}/data/**/*.parquet", low_memory=True)
#         .select("subject_id", "time", "code", "numeric_value", "text_value")
#         .join(sample.lazy(), on="subject_id", how="inner")
#         .filter(before_prediction_time)
#         .with_columns(remove_long_text)
#     )

#     top_99_codes = (
#         events_base.select("code")
#         .group_by("code")
#         .agg(pl.len().alias("count"))
#         .sort(pl.col("count"), descending=True)
#         # .filter(pl.col("count") > pl.col("count").quantile(0.995))
#         .with_columns((pl.col("count") / n_patients).alias("count_div"))
#         .filter(pl.col("count_div") >= 13)
#         .collect(engine="streaming")
#     )

#     events_with_ids = (
#         events_base.join(top_99_codes.lazy(), on="code", how="anti")
#         .with_columns(parsed=parse_txt_to_float)
#         .with_columns(numeric_value=swap_text_with_numeric[0])
#         .with_columns(text_value=swap_text_with_numeric[1])
#         # keep original code/text for grouping after explosion
#         .with_columns(orig_code=pl.col("code"), orig_text=pl.col("text_value"))
#         # create the list-of-values to explode
#         .with_columns(bp_values=bp_values_list)
#         # explode -> if BP valid you'll get 2 rows, otherwise 1 row
#         .explode("bp_values")
#         # set the numeric_value from the exploded element
#         .with_columns(numeric_value=pl.col("bp_values"))
#         # assign new code names for BP rows using a windowed max over the original group
#         .with_columns(
#             code=pl.when(pl.col("orig_code").str.contains("Blood Pressure"))
#             .then(
#                 pl.when(
#                     pl.col("numeric_value")
#                     == pl.col("bp_values")
#                     .max()
#                     .over(["subject_id", "time", "orig_text"])
#                 )
#                 .then(pl.lit("Systolic blood pressure"))
#                 .otherwise(pl.lit("Diastolic blood pressure"))
#             )
#             .otherwise(pl.col("code"))
#         )
#         # set text_value to null for the BP-derived rows, keep other text_value unchanged
#         .with_columns(
#             text_value=pl.when(pl.col("orig_code").str.contains("Blood Pressure"))
#             .then(pl.lit(None).cast(pl.Utf8))
#             .otherwise(pl.col("text_value"))
#         )
#         # cleanup helper columns
#         .drop(["bp_values", "orig_code", "orig_text", "parsed"])
#         .with_columns(regenerate_ids)
#         .collect(engine="streaming")
#     )

#     # #print(events_with_ids.select(pl.len()).collect().item())

#     joblib.dump(
#         value=(
#             events_with_ids.select(["subject_id", "boolean_value"])
#             .group_by("subject_id")
#             .agg(pl.col("boolean_value").first().cast(pl.Int8).alias("boolean_value"))
#             .sort("subject_id")
#             .select("boolean_value")
#             .to_series()
#             .to_list()
#         ),
#         filename=f"{LABELS_DIR}/outcomes_meds_{TIME_OPT}_{n_patients}_{idx}.joblib",
#     )

#     events_with_ids.select(
#         ["subject_id", "code", "time", "numeric_value", "text_value"]
#     ).write_parquet(f"{TMP_DATA_DIR}/0.parquet")

#     MedsRDFConverter(TMP_DIR).convert(
#         sink=NTriplesSink(
#             Path(f"{EXPORT_DIR}/meds_{n_patients}_{idx}"), gzip_mode=True
#         ),
#         cfg=Config(schemas={MEDSSchema.CODES}),
#     )

import os

MIMIC_ETL_OUTPUT = "MIMIC/MEDS_cohort"
MIMIC_ETL_GRAPH = f"{MIMIC_ETL_OUTPUT}/graph"
MIMIC_TASKS_PATH = "MIMIC/tasks"
TIME_OPT = "TS"

os.makedirs(MIMIC_ETL_GRAPH, exist_ok=True)

import os
import polars as pl

import joblib

from meds2rdf import MedsRDFConverter
from meds2rdf.sinks import NTriplesSink
from meds2rdf.config import Config, MEDSSchema
from pathlib import Path

TMP_DIR = f"{MIMIC_ETL_OUTPUT}/tmp"
TMP_DATA_DIR = f"{TMP_DIR}/data/train"
TMP_METADATA_DIR = f"{TMP_DIR}/metadata"

os.makedirs(TMP_DATA_DIR, exist_ok=True)
os.makedirs(TMP_METADATA_DIR, exist_ok=True)
os.makedirs(MIMIC_ETL_GRAPH, exist_ok=True)

NUM_PATIENTS = 1800
N_FOLDS = 5

# long_term_reccurrence
# inhospital_mortality (# 3_329 M 2_666 T, # 62_376 M 76_782 F)
# imminent_mortality (# 125497 T  79353 F)

# los_in_hospital_24h
# los_in_icu_24h
# mortality_in_hospital_48h
# mortality_in_icu_48h
# readmission_30d_in_icu_48h
# readmission_30d_in_hospital_48h
TASKS = ["los_in_hospital_24h", "mortality_in_hospital_48h", "mortality_in_icu_48h", "readmission_30d_in_icu_48h", "readmission_30d_in_hospital_48h"]

for TASK in TASKS:
    IHM = f"{MIMIC_ETL_OUTPUT}/labels/{TASK}/**/*.parquet"
    EXPORT_DIR = f"exports/{TASK}"
    LABELS_DIR = f"{EXPORT_DIR}/labels"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    from MIMIC.utils2 import generate_samples

    outcomes = pl.scan_parquet(IHM).select("subject_id", "prediction_time", "boolean_value")

    samples = generate_samples(
        labels=outcomes, n_folds=N_FOLDS, size=NUM_PATIENTS, seed=1234
    )

    from MIMIC.utils2 import *

    REMOVE_OCCURRENCES = 13

    events = pl.scan_parquet(
        f"{MIMIC_ETL_OUTPUT}/data/**/*.parquet", low_memory=True
    ).select("subject_id", "time", "code", "numeric_value", "text_value")

    for idx, sample in enumerate(samples):
        events_base = (
            events.join(sample.lazy(), on="subject_id", how="inner")
            .filter(before_prediction_time)
            .with_columns(remove_long_text)
        )

        most_frequent_codes = (
            events_base.select("code")
            .group_by("code")
            .agg(pl.len().alias("count"))
            .sort(pl.col("count"), descending=True)
            # .filter(pl.col("count") > pl.col("count").quantile(0.995))
            .with_columns((pl.col("count") / NUM_PATIENTS).alias("count_div"))
            .filter(pl.col("count_div") >= REMOVE_OCCURRENCES)
            .collect(engine="streaming")
        )

        events_with_ids = (
            events_base.join(most_frequent_codes.lazy(), on="code", how="anti")
            .with_columns(parsed=parse_txt_to_float)
            .with_columns(numeric_value=swap_text_with_numeric[0])
            .with_columns(text_value=swap_text_with_numeric[1])
            # keep original code/text for grouping after explosion
            .with_columns(orig_code=pl.col("code"), orig_text=pl.col("text_value"))
            # create the list-of-values to explode
            .with_columns(bp_values=bp_values_list)
            # explode -> if BP valid you'll get 2 rows, otherwise 1 row
            .explode("bp_values")
            # set the numeric_value from the exploded element
            .with_columns(numeric_value=pl.col("bp_values"))
            # assign new code names for BP rows using a windowed max over the original group
            .with_columns(code=to_systolic_and_dyastolic_pressure)
            # set text_value to null for the BP-derived rows, keep other text_value unchanged
            .with_columns(text_value=set_BP_values_text_to_null)
            # cleanup helper columns
            .drop(derived_columns)
            .with_columns(regenerate_ids)
            .collect(engine="streaming")
        )

        joblib.dump(
            value=extract_labels_array(events_with_ids),
            filename=f"{LABELS_DIR}/outcomes_meds_{TIME_OPT}_{NUM_PATIENTS}_{idx}.joblib",
        )

        events_with_ids.select(
            ["subject_id", "code", "time", "numeric_value", "text_value"]
        ).write_parquet(f"{TMP_DATA_DIR}/0.parquet")

        MedsRDFConverter(TMP_DIR).convert(
            sink=NTriplesSink(
                Path(f"{EXPORT_DIR}/meds_{NUM_PATIENTS}_{idx}"), gzip_mode=True
            ),
            cfg=Config(schemas={MEDSSchema.CODES}),
        )