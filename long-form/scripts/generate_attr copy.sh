#!/bin/bash

set -x;
# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --method base

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path results/attribution/qwen3-32b_gov_report_validation_300_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path results/attribution/qwen3-32b_gov_report_validation_300_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix


# python generate_attr_summary.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 4000 --num_sents 30

# python generate_attr_summary.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 4000 --num_sents 50

python generate_attr_summary.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 15000 --num_sents 70

# python generate_attr_summary.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 4000 --num_sents 30

python generate_attr_summary.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 10000 --num_sents 50

python generate_attr_summary.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 15000 --num_sents 70

python generate_attr_summary.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 5000 --num_sents 30

python generate_attr_summary.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 10000 --num_sents 50

python generate_attr_summary.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 15000 --num_sents 70
# Extract generative attribution on QMSum

# python generate_attr_summary.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 50 --max-tokens 5000 --num_sents 30

# python generate_attr_summary.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 50 --max-tokens 10000 --num_sents 50

# python generate_attr_summary.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 50 --max-tokens 15000 --num_sents 70

# python generate_attr_summary.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 50 --max-tokens 5000 --num_sents 30

# python generate_attr_summary.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 50 --max-tokens 10000 --num_sents 50

# python generate_attr_summary.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 50 --max-tokens 15000 --num_sents 70

# python generate_attr_summary.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 50 --max-tokens 5000 --num_sents 30

# python generate_attr_summary.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 50 --max-tokens 10000 --num_sents 50

# python generate_attr_summary.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 50 --max-tokens 15000 --num_sents 70