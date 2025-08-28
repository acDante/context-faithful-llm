#!/bin/bash

DATASET=${1:-"gov_report"}
MODEL=${2:-"llama3.1-8b"}

NUM_SAMPLES=${3:-"300"}
NUM_SENTS=${4:-"30"}

data_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/summary"
attr_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/attribution"

JSON_FILES=(
    "${attr_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_gen_attr_sentence_num${NUM_SENTS}.json"
    "${data_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_base.json"
    "${data_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_attr-sent${NUM_SENTS}_base+impt.json"
    "${data_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_attr-sent${NUM_SENTS}_base+impt_prefix.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    LOG_FILE="${JSON_FILE%.json}.log"

    DATA_PATH="${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python eval_mint.py --data_path $DATA_PATH \
                             --dataset $DATASET \
                             --batch_size 1
    fi
done