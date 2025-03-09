#!/bin/bash

# Directory containing the prediction files
DIRECTORY="/home/xiaotang/Project/context-faithful-llm/guided-cad/results/summary"

# Iterate over models
MODELS=("mistral-7b")
# MODELS=("mistral-7b" "llama3-8b")

# Iterate over attribution methods
ATTRIBUTION_METHODS=("attention")
# ATTRIBUTION_METHODS=("attention" "saliency")

# Iterate over experiment settings
ALPHAS=(-0.5 0.5 1.0 1.5 2.0)

for model in "${MODELS[@]}"; do
    for attribution_method in "${ATTRIBUTION_METHODS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            DATA_PATH="${DIRECTORY}/ccsum/cad/ccsum-mistral-7b-attention-impt+cad_v2-alpha_${alpha}_preds.json"
            echo "Evaluating: ${DATA_PATH}"
            # DATA_PATH="${DIRECTORY}/xsum-${model}-${attribution_method}-impt+cad-alpha_${alpha}_preds.json"
            # Evaluate ROUGE, BERT scores and Summa-C scores
            if [ -f "$DATA_PATH" ]; then
                # python eval_basic.py --data_path ${DATA_PATH} \
                #                      --dataset xsum \
                #                      --log_path results/xsum-${model}-tune-alpha.log
                # python eval_basic.py --data_path $DATA_PATH --dataset ccsum --log_path results/ccsum-${model}-tune-alpha.log
                python eval_prefs.py --data_path $DATA_PATH --dataset ccsum --metrics factscore --log_path results/ccsum-${model}-factscore.log
            fi
        done
    done
done