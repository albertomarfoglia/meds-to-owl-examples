#!/usr/bin/env bash

set -e  # stop on error

# -----------------------
# Variables (from your Python)
# -----------------------
MIMIC_ETL_OUTPUT="MIMIC/MEDS_cohort"
MIMIC_ETL_GRAPH="${MIMIC_ETL_OUTPUT}/graph"
MIMIC_TASKS_PATH="MIMIC/tasks"
TIME_OPT="TS"

TMP_DIR="${MIMIC_ETL_OUTPUT}/tmp"
TMP_DATA_DIR="${TMP_DIR}/data/train"
TMP_METADATA_DIR="${TMP_DIR}/metadata"

export HYDRA_FULL_ERROR=1

# -----------------------
# Main loop
# -----------------------
for f in "$MIMIC_TASKS_PATH"/*.yaml; do
    filename=$(basename -- "$f")
    f_name="${filename%.*}"

    echo "Running task: $f_name"

    aces-cli \
        config_path="$f" \
        cohort_name="$f_name" \
        cohort_dir="$MIMIC_ETL_OUTPUT/labels" \
        data=sharded \
        data.standard=meds \
        data.root="$MIMIC_ETL_OUTPUT/data" \
        data.shard=$(expand_shards train/292 tuning/37 held_out/37) \
        -m
done