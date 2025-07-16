#!/bin/bash
# Evaluate ROUGE-L, BERTScore and SummaC for a list of prediction files

DATASET=${1:-"xsum"}
MODEL=${2:-"llama3.1-8b"}
ATTRIBUTION=${3:-"attention"}

echo "Running experiments on dataset: $DATASET"

# Directory containing the prediction files
DIRECTORY="/home/xiaotang/Project/context-faithful-llm/guided-cad/results/summary/${DATASET}/${MODEL}"
# DIRECTORY="/mnt/ceph_rbd/project/context-faithful-llm/guided-cad/results/summary/"
if [ ! -d "$DIRECTORY" ]; then
  echo "Error: result path $DIRECTORY not found"
  exit 1
fi

JSON_FILES=(
    "${DATASET}-${MODEL}-base_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-1000_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt_prefix-1000_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt-1000_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt-cad-1000_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-impt_only-1000_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-mask_impt-cad-1000_preds.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    LOG_FILE="${JSON_FILE%.json}.log"

    DATA_PATH="${DIRECTORY}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python new_eval_basic.py --data_path $DATA_PATH \
                                 --dataset ${DATASET}
    fi
done