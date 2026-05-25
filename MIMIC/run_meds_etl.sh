#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Default output directory
OUTPUT_DIR="${1:-.}"
DATASET_USERNAME="${2:-}"
DATASET_PASSWORD="${3:-}"
DO_DOWNLOAD="${4:-False}"

# Create virtual environment if it doesn't exist
# if [ ! -d ".venv_meds_mimic" ]; then
#     conda create -y -n venv_meds_mimic python=3.11
#     conda activate venv_meds_mimic
# fi

export DATASET_DOWNLOAD_USERNAME="${DATASET_USERNAME}"
export DATASET_DOWNLOAD_PASSWORD="${DATASET_PASSWORD}"
export HYDRA_FULL_ERROR=1
export N_WORKERS=6

# Upgrade pip (optional but recommended)
pip install --upgrade pip

pip install hydra-joblib-launcher

# Install required package
pip install MIMIC_IV_MEDS

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run MEDS extraction
MEDS_extract-MIMIC_IV \
    root_output_dir="${OUTPUT_DIR}" \
    do_download="${DO_DOWNLOAD}"
    #download_workers="8" \
    #do_overwrite=False \
    #do_copy=True \

echo "Extraction completed. Output saved to '${OUTPUT_DIR}'"
