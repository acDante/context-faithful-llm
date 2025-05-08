#!/bin/bash

# Run for XSum dataset
python attention.py --dataset xsum --model_name meta-llama/Llama-3.1-8B-Instruct --num_samples 1000 --num_sents 3 --save_path results/xsum-llama3.1-8b-attention-1000.json --mode last

# Run for your second dataset (replace dataset_name with your second dataset)
# python attention.py --dataset ccsum --model_name meta-llama/Llama-3.1-8B-Instruct --num_samples 1000 --num_sents 3 --save_path results/ccsum-llama3.1-8b-attention-1000.json --mode last
