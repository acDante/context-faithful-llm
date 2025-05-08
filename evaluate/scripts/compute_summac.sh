#!/bin/bash
# Recompute FactScore for the predictions on CNN/DM dataset

DATASET="$1"
MODEL="$2"
ATTRIBUTION="attention"

echo "Running experiments on dataset: $DATASET"

# Directory containing the prediction files
# DIRECTORY="/mnt/ceph_rbd/project/context-faithful-llm/guided-cad/results/summary/${DATASET}/${MODEL}"
DIRECTORY="/mnt/ceph_rbd/project/context-faithful-llm/guided-cad/results/summary/${MODEL}"
if [ ! -d "$DIRECTORY" ]; then
  echo "Error: result path $DIRECTORY not found"
  exit 1
fi

JSON_FILES=(
    # "${DATASET}-${MODEL}-${ATTRIBUTION}-base_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-impt_only_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-impt+cad_preds.json"
)


for JSON_FILE in "${JSON_FILES[@]}"; do
    LOG_FILE="${JSON_FILE%.json}.log"

    DATA_PATH="${DIRECTORY}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_basic.py --data_path $DATA_PATH \
                             --dataset ${DATASET} \
                             --log_path logs/$LOG_FILE
    fi
done