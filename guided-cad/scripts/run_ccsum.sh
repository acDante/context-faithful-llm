#!/bin/bash
# Script for generating summary on CCSum

# Activate conda env
source ~/.bashrc
conda activate llm

attribution_methods=("attention")
schema_methods=("base+impt" "impt_only" "base+impt_prefix")

for attribution_method in "${attribution_methods[@]}"; do
    for schema in "${schema_methods[@]}"; do
        # Experiments with mistral-7b
        # python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 \
        #                            --dataset ccsum \
        #                            --attr_data_path results/ccsum-mistral-7b-${attribution_method}-1000.json \
        #                            --num_samples 1000 \
        #                            --log_path results/summary/ccsum \
        #                            --exp_name ccsum-mistral-7b-${attribution_method}-${schema} \
        #                            --schema ${schema}

        # Experiments with Llama3-8b
        # python generate_summary.py --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        #                            --dataset ccsum \
        #                            --attr_data_path results/ccsum-llama3-8b-${attribution_method}-1000.json \
        #                            --num_samples 1000 \
        #                            --log_path results/summary/ccsum/llama3 \
        #                            --exp_name ccsum-llama3-8b-${attribution_method}-${schema} \
        #                            --schema ${schema}
        
        # Baseline experiments with Llama3.1-8b
        set -x;
        python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct \
                                       --dataset ccsum \
                                       --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json \
                                       --num_samples 1000 \
                                       --log_path results/summary/ccsum/llama3.1-8b \
                                       --exp_name ccsum-llama3.1-8b-attention-${schema} \
                                       --schema ${schema}

    done
done

# CAD experiment with Llama3.1-8b
# python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct \
#                                 --dataset ccsum \
#                                 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json \
#                                 --num_samples 1000 \
#                                 --log_path results/summary/ccsum/llama3.1-8b \
#                                 --exp_name ccsum-llama3.1-8b-attention-impt+cad \
#                                 --schema base+impt \
#                                 --method cad \
#                                 --alpha 0.5

# python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct \
#                                 --dataset ccsum \
#                                 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json \
#                                 --num_samples 1000 \
#                                 --log_path results/summary/ccsum/llama3.1-8b \
#                                 --exp_name ccsum-llama3.1-8b-attention-mask_impt+cad \
#                                 --schema mask_impt \
#                                 --method cad \
#                                 --alpha 0.5