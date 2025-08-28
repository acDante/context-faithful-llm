#!/bin/bash

datasets=("gov_report" "qmsum")
models=("llama3.1-8b" "qwen3-8b" "qwen3-32b")

# Compute all the evaluation metrics for the SumCoT experiments on GovReport and QMSum
data_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/summary"

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Set number of samples based on dataset
        if [ "$dataset" = "gov_report" ]; then
            num_samples=300
        elif [ "$dataset" = "qmsum" ]; then
            num_samples=272
        elif [ "$dataset" = "summscreen" ]; then
            num_samples=338
        else
            echo "Error: Unknown dataset '$dataset'"
            continue
        fi
        
        JSON_FILE="${data_path}/${model}_${dataset}_validation_${num_samples}_sum_cot.json"
        # Add your processing logic here
        DATA_PATH="${JSON_FILE}"
        if [ -f "$DATA_PATH" ]; then
            echo "Evaluating ${DATA_PATH}"
            python eval_prefs.py --data_path "$DATA_PATH" \
                                     --dataset ${dataset} \
                                     --metrics factscore \
                                     --model_name gpt-4o-mini
        fi
    done
done