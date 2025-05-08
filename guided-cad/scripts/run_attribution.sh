#!/bin/bash

# Activate conda env
source ~/.bashrc
conda activate inseq

# Try different attribution methods
# model=("")
attribution_methods=("attention" "saliency")

for attribution_method in "${attribution_methods[@]}"
do
    python extract_attr_inseq.py --dataset ccsum \
                                 --model_name mistralai/Mistral-7B-Instruct-v0.2 \
                                 --attribution ${attribution_method} \
                                 --num_samples 1000 \
                                 --num_sents 3 \
                                 --save_path results/ccsum-mistral-7b-${attribution_method}-1000.json
done

# Extract attention attribution using Llama3.1-8b model
python extract_attr_inseq.py --dataset cnn_dm --model_name meta-llama/Llama-3.1-8B-Instruct --attribution attention --num_samples 1000 --num_sents 3 --save_path results/attribution/cnn_dm-llama3.1-8b-attention-1000.json
python extract_attr_inseq.py --dataset xsum --model_name meta-llama/Llama-3.1-8B-Instruct --attribution attention --num_samples 1000 --num_sents 3 --save_path results/attribution/xsum-llama3.1-8b-attention-1000.json
python extract_attr_inseq.py --dataset ccsum --model_name meta-llama/Llama-3.1-8B-Instruct --attribution attention --num_samples 1000 --num_sents 3 --save_path results/attribution/ccsum-llama3.1-8b-attention-1000.json