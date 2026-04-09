#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Default output directory
OUTPUT_DIR="${1:-MIMIC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv_mimic" ]; then
    python3 -m venv .venv_mimic
fi

export DATASET_DOWNLOAD_USERNAME=""
export DATASET_DOWNLOAD_PASSWORD=""
export HYDRA_FULL_ERROR=1
export N_WORKERS=1

# Activate virtual environment
source .venv_mimic/bin/activate

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
    do_download=False \
    do_overwrite=True

#    do_copy=True \

echo "Extraction completed. Output saved to '${OUTPUT_DIR}'"
