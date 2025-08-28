#!/bin/bash
# Evaluation script for the ablation study on the number of attributed sentences

NUM_SENTS=${1:-"30"}

ROOT_PATH="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/ablation"
DATA_PATH="${ROOT_PATH}/qwen3-8b_gov_report_train_100_attr-sent${NUM_SENTS}_base+impt.json"

if [ -f "$DATA_PATH" ]; then
    echo "Evaluating ${DATA_PATH}"
    set -x;
    python new_eval_basic.py --data_path "$DATA_PATH" \
                             --dataset gov_report
    
    python eval_prefs.py --data_path $DATA_PATH \
                            --dataset gov_report \
                            --metrics factscore \
                            --model_name gpt-4o-mini
fi