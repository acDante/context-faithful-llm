#!/bin/bash
# Script for evaluation of ROUGE-L, BERTScore, SummaC for experiments on GovReport

DATASET=${1:-"gov_report"}
MODEL=${2:-"llama3.1-8b"}

NUM_SAMPLES=${3:-"338"}
NUM_SENTS=${4:-"50"}

data_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/summary"
attr_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/attribution"

JSON_FILES=(
    "${attr_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_gen_attr_sentence_num${NUM_SENTS}.json"
    # "${data_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_base.json"
    "${data_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_attr-sent${NUM_SENTS}_base+impt.json"
    "${data_path}/${MODEL}_${DATASET}_validation_${NUM_SAMPLES}_attr-sent${NUM_SENTS}_base+impt_prefix.json"
)

for JSON_FILE in "${JSON_FILES[@]}"; do
    LOG_FILE="${JSON_FILE%.json}.log"

    DATA_PATH="${JSON_FILE}"
    if [ -f "$DATA_PATH" ]; then
        echo "Evaluating ${DATA_PATH}"
        python new_eval_basic.py --data_path $DATA_PATH \
                                 --dataset ${DATASET}
    fi
done