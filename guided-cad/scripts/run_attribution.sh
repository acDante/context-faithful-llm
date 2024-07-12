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