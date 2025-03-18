#!/bin/bash
# Recompute FactScore for the predictions on CCSum dataset

# Directory containing the prediction files
DIRECTORY="/home/xiaotang/Project/context-faithful-llm/guided-cad/results/summary/ccsum"

JSON_FILES=(
    "ccsum-mistral-7b-attention-base_preds.json"
    "ccsum-mistral-7b-attention-impt_only_preds.json"
    "ccsum-mistral-7b-attention-base+impt_preds.json"
    "ccsum-mistral-7b-attention-impt+cad_preds.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    DATA_PATH="${DIRECTORY}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_prefs.py --data_path $DATA_PATH \
                            --dataset ccsum \
                            --metrics factscore \
                            --model_name gpt-4o-mini
    fi
done