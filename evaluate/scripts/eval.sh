#!/bin/bash

# Directory containing the JSON files
DIRECTORY="/home/xiaotang/Project/context-faithful-llm/guided-cad/results/summary"

# Iterate over models
# MODELS=("llama3-8b" "mistral-7b")
MODELS=("mistral-7b")

# Iterate over attribution methods
ATTRIBUTION_METHODS=("attention" "saliency")

# Iterate over experiment settings
SETTINGS=("base" "base+impt" "impt_only")

# for model in "${MODELS[@]}"; do
#     # BASE_PATH="${DIRECTORY}/${model}/xsum-${model}-base_preds.json"
#     BASE_PATH="${DIRECTORY}/ccsum/ccsum-mistral-7b-attention-base_preds.json"
#     if [ -f "$BASE_PATH" ]; then
#         python eval_basic.py --data_path $BASE_PATH --dataset ccsum --log_path results/ccsum-mistral-7b-experiments.log
#         # python eval_prefs.py --data_path $BASE_PATH --dataset ccsum --metrics factscore --log_path results/ccsum-mistral-7b-experiments.log
#      #   python eval_basic.py --data_path $BASE_PATH --dataset xsum --log_path results/xsum-${model}-experiments.log
#         # python eval_prefs.py --data_path $BASE_PATH --metrics factscore --log_path eval.log
#     fi
#     for attribution_method in "${ATTRIBUTION_METHODS[@]}"; do
#         for setting in "${SETTINGS[@]}"; do
#             # Evaluate FactScores
#             DATA_PATH="${DIRECTORY}/ccsum/ccsum-mistral-7b-${attribution_method}-${setting}_preds.json"
#             # DATA_PATH="${DIRECTORY}/${model}/xsum-${model}-${attribution_method}-${setting}_preds.json"
#             if [ -f "$DATA_PATH" ]; then
#                 python eval_basic.py --data_path $DATA_PATH --dataset ccsum --log_path results/ccsum-mistral-7b-experiments.log
#                 # python eval_prefs.py --data_path $DATA_PATH --dataset ccsum --metrics factscore --log_path results/ccsum-mistral-7b-experiments.log
#             fi
#         done
#     done
# done

for model in "${MODELS[@]}"; do
    for attribution_method in "${ATTRIBUTION_METHODS[@]}"; do
        for setting in "${SETTINGS[@]}"; do
            # Evaluate FactScores
            DATA_PATH="${DIRECTORY}/ccsum/ccsum-mistral-7b-${attribution_method}-${setting}_preds.json"
            # DATA_PATH="${DIRECTORY}/${model}/xsum-${model}-${attribution_method}-${setting}_preds.json"
            if [ -f "$DATA_PATH" ]; then
                # python eval_basic.py --data_path $DATA_PATH --dataset ccsum --log_path results/ccsum-mistral-7b-experiments.log
                python eval_prefs.py --data_path $DATA_PATH --dataset ccsum --metrics factscore --log_path results/ccsum-mistral-7b-experiments-factscore.log
            fi
        done
    done
done