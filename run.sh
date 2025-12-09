rm -rf output

MEDS_transform-pipeline \
    pkg://MEDS_extract.configs._extract.yaml \
    --overrides \
    input_dir=intermediate \
    output_dir=output \
    event_conversion_config_fp=MESSY.yaml \
    dataset.name=Neurovasc \
    dataset.version=1.0