#!/bin/bash
# Evaluate faithfulness using Qwen model as annotator

DATASET=${1:-"xsum"}
MODEL=${2:-"llama3.1-8b"}
ATTRIBUTION=${3:-"attention"}

echo "Running experiments on dataset: $DATASET"

# Directory containing the prediction files
DIRECTORY="/mnt/ceph_rbd/project/context-faithful-llm/guided-cad/results/summary/${DATASET}/${MODEL}"

if [ ! -d "$DIRECTORY" ]; then
  echo "Error: result path $DIRECTORY not found"
  exit 1
fi

JSON_FILES=(
    "${DATASET}-${MODEL}-base_preds.json"
    "${DATASET}-${MODEL}-base-cad_preds.json"
    "${DATASET}-${MODEL}-base-dola-low_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt_prefix_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt-cad_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-impt_only_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-mask_impt-cad_preds.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    DATA_PATH="${DIRECTORY}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_with_llm.py --data_path $DATA_PATH \
                                --dataset ${DATASET} \
                                --model_name Qwen/Qwen3-32B \
    else
        echo "Warning: File $DATA_PATH does not exist, skipping."
    fi
done