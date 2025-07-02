#!/bin/bash
# Evaluate percentage of novel n-gram

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

# # for llama3.1-8b on CNN/DM
# JSON_FILES=(
#     "${DATASET}-${MODEL}-${ATTRIBUTION}-base-1000_preds.json"
#     "${DATASET}-${MODEL}-base-cad-1000_preds.json"
#     "${DATASET}-${MODEL}-base-dola-low-1000_preds.json"
#     # "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt_prefix_preds.json"
#     "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt-1000_preds.json"
#     "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt-cad-1000_preds.json"
#     "${DATASET}-${MODEL}-${ATTRIBUTION}-impt_only-1000_preds.json"
#     # "${DATASET}-${MODEL}-${ATTRIBUTION}-mask_impt-cad_preds.json"
# )

JSON_FILES=(
    "${DATASET}-${MODEL}-base_preds.json"
    "${DATASET}-${MODEL}-base-cad_preds.json"
    "${DATASET}-${MODEL}-base-dola-low_preds.json"  
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt_preds.json"
    "${DATASET}-${MODEL}-${ATTRIBUTION}-base+impt-cad_preds.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    DATA_PATH="${DIRECTORY}/${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_ngram.py --data_path $DATA_PATH \
                                --dataset ${DATASET} \
                                --max_ngram_size 3

    else
        echo "Warning: File $DATA_PATH does not exist, skipping."
    fi
done