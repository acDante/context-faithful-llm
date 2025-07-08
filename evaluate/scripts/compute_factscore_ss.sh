#!/bin/bash

data_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/summary"
attr_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/attribution"

JSON_FILES=(
    "qwen3-32b_summscreen_validation_338.json"
    "qwen3-32b_summscreen_validation_338_attr-fact-5_base+impt.json"
    "qwen3-32b_summscreen_validation_338_attr-sentence-5_base+impt.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    DATA_PATH="${data_path}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_prefs.py --data_path $DATA_PATH \
                            --dataset summscreen \
                            --metrics factscore \
                            --model_name gpt-4o-mini
    fi
done

JSON_FILES=(
    "qwen3-32b_summscreen_validation_338_gen_attr_fact_num5.json"
    "qwen3-32b_summscreen_validation_338_gen_attr_sentence_num5.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    DATA_PATH="${attr_path}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_prefs.py --data_path $DATA_PATH \
                            --dataset summscreen \
                            --metrics factscore \
                            --model_name gpt-4o-mini
    fi
done