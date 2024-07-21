#!/bin/bash

# Activate conda env
source ~/.bashrc
conda activate llm

# Test different alpha values
# alpha_values=(-0.5 0.5 1.0 1.5 2.0)
alpha_values=(-0.5 0.5)
# alpha_values=(1.0 1.5 2.0)

# Run CAD experiments on XSum
# for alpha in "${alpha_values[@]}"; do
#     echo "Running attribution-guided contrastive decoding with --alpha=${alpha}"
#     python generate_summary.py --model_name meta-llama/Meta-Llama-3-8B-Instruct \
#                                --dataset xsum \
#                                --attr_data_path results/xsum-llama3-8b-saliency-1000.json \
#                                --num_samples 1000 \
#                                --exp_name xsum-llama3-8b-saliency-impt+cad-alpha_${alpha} \
#                                --schema base+impt --use_cad --alpha ${alpha}
# done

# Run CAD experiments on XSum (with new contrastive setting)
attribution_methods=("attention")
for attribution_method in "${attribution_methods[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        echo "Running attribution-guided contrastive decoding with --alpha=${alpha}"
        set -x;
        python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 \
                                --dataset xsum \
                                --attr_data_path results/xsum-mistral-7b-${attribution_method}-1000.json \
                                --num_samples 1000 \
                                --log_path results/summary/xsum \
                                --exp_name xsum-mistral-7b-${attribution_method}-impt+cad_v2-alpha_${alpha} \
                                --schema base+impt_v2 --use_cad --alpha ${alpha}
    done
done

# Run CAD experiments on CCSum
# attribution_methods=("attention" "saliency")
# for attribution_method in "${attribution_methods[@]}"; do
#     for alpha in "${alpha_values[@]}"; do
#         echo "Running attribution-guided contrastive decoding with --alpha=${alpha}"
#         python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 \
#                                 --dataset ccsum \
#                                 --attr_data_path results/ccsum-mistral-7b-${attribution_method}-1000.json \
#                                 --num_samples 1000 \
#                                 --log_path results/summary/ccsum/cad \
#                                 --exp_name ccsum-mistral-7b-${attribution_method}-impt+cad-alpha_${alpha} \
#                                 --schema base+impt --use_cad --alpha ${alpha}
#     done
# done

# Run CAD experiments on CCSum (with new contrastive setting)
# attribution_methods=("attention")
# for attribution_method in "${attribution_methods[@]}"; do
#     for alpha in "${alpha_values[@]}"; do
#         echo "Running attribution-guided contrastive decoding with --alpha=${alpha}"
#         set -x;
#         python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 \
#                                 --dataset ccsum \
#                                 --attr_data_path results/ccsum-mistral-7b-${attribution_method}-1000.json \
#                                 --num_samples 1000 \
#                                 --log_path results/summary/ccsum/cad \
#                                 --exp_name ccsum-mistral-7b-${attribution_method}-impt+cad_v2-alpha_${alpha} \
#                                 --schema base+impt_v2 --use_cad --alpha ${alpha}
#     done
# done