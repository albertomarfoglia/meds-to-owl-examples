#!/usr/bin/env bash

FILE="MIMIC/experiments.yaml"

# Loop over all tasks
yq -r '.experiments[][] | [.task, .num_of_samples, .agg] | @tsv' "$FILE" | while IFS=$'\t' read -r task num_of_samples agg; do
    echo "Task: $task"
    
    # Loop over num_of_samples
    for ((i=0; i<num_of_samples; i++)); do

        BASE_DIR="exports/$task/meds/$i/MEDS_cohort"

        RESHARD_DIR="$BASE_DIR/reshard"
        DATA_DIR="$RESHARD_DIR/data"
        OUTPUT_DIR="$BASE_DIR/output"
        MODEL_DIR="$BASE_DIR/output_model/first_24_in_hospital_mortality"
        LABELS_DIR="$BASE_DIR/labels"
        WINDOW=$agg

        MEDS_transform-reshard_to_split \
        --multirun \
        worker="range(0,6)" \
        hydra/launcher=joblib \
        input_dir="$BASE_DIR" \
        cohort_dir="$RESHARD_DIR" \
        'stages=["reshard_to_split"]' \
        stage="reshard_to_split" \
        stage_configs.reshard_to_split.n_subjects_per_shard=2500

        meds-tab-describe \
            "input_dir=$DATA_DIR" \
            "output_dir=$OUTPUT_DIR"
            
        meds-tab-tabularize-static \
            "input_dir=$DATA_DIR" \
            "output_dir=$OUTPUT_DIR" \
            tabularization.min_code_inclusion_count=10 \
            tabularization.window_sizes=[$WINDOW] \
            do_overwrite=False \
            tabularization.aggs=[static/present,static/first]
            #tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]

        meds-tab-tabularize-time-series \
            --multirun \
            worker="range(0,6)" \
            hydra/launcher=joblib \
            "input_dir=$DATA_DIR" \
            "output_dir=$OUTPUT_DIR" \
            tabularization.min_code_inclusion_count=10 \
            tabularization.window_sizes=[$WINDOW] \
            tabularization.aggs=[static/present,static/first]

        meds-tab-cache-task \
            --multirun \
            hydra/launcher=joblib \
            worker="range(0,6)" \
            "input_dir=$DATA_DIR" \
            "output_dir=$OUTPUT_DIR" \
            "input_label_dir=$LABELS_DIR" \
            "task_name=$task" \
            tabularization.min_code_inclusion_count=10 \
            tabularization.window_sizes=[$WINDOW] \
            tabularization.aggs=[static/present,static/first]

        meds-tab-model \
            --multirun \
            model_launcher=xgboost \
            "input_dir=$DATA_DIR" \
            "output_dir=$OUTPUT_DIR" \
            "output_model_dir=$MODEL_DIR" \
            "task_name=$task" \
            "hydra.sweeper.n_trials=100" \
            "hydra.sweeper.n_jobs=6" \
            tabularization.min_code_inclusion_count=10 \
            tabularization.window_sizes=[$WINDOW] \
            tabularization.aggs=[static/present,static/first]

    done
done

exit 0


# code/count aumenta di parecchio la prevsione