#!/bin/bash

# Activate conda env
source ~/.bashrc
conda activate llm

# Test different alpha values
alpha_values=(-0.5 0.5 1.0 1.5 2.0)

# Run experiments on XSum
# for alpha in "${alpha_values[@]}"; do
#     echo "Running attribution-guided contrastive decoding with --alpha=${alpha}"
#     python generate_summary.py --model_name meta-llama/Meta-Llama-3-8B-Instruct \
#                                --dataset xsum \
#                                --attr_data_path results/xsum-llama3-8b-saliency-1000.json \
#                                --num_samples 1000 \
#                                --exp_name xsum-llama3-8b-saliency-impt+cad-alpha_${alpha} \
#                                --schema base+impt --use_cad --alpha ${alpha}
# done

# Run experiments on CCSum
attribution_methods=("attention" "saliency")
for attribution_method in "${attribution_methods[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        echo "Running attribution-guided contrastive decoding with --alpha=${alpha}"
        python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 \
                                --dataset ccsum \
                                --attr_data_path results/ccsum-mistral-7b-${attribution_method}-1000.json \
                                --num_samples 1000 \
                                --log_path results/summary/ccsum/cad \
                                --exp_name ccsum-mistral-7b-${attribution_method}-impt+cad-alpha_${alpha} \
                                --schema base+impt --use_cad --alpha ${alpha}
    done
done