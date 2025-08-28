#!/bin/bash

# Run SumCoT baseline on GovReport
python summary_cot.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gov_report --max-samples 300 --max-tokens-sum 1024 --max-model-len 85000 --gpu-memory-utilization 0.9

python summary_cot.py --model Qwen/Qwen3-8B --dataset gov_report --max-samples 300 --max-tokens-sum 1024 --max-model-len 85000 --gpu-memory-utilization 0.9

python summary_cot.py --model Qwen/Qwen3-32B --dataset gov_report --max-samples 300 --max-tokens-sum 1024 --max-model-len 85000 --gpu-memory-utilization 0.9

# Run SumCoT baseline on QMSum
python summary_cot.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 300 --max-tokens-sum 512 --max-model-len 40000

python summary_cot.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 300 --max-tokens-sum 512 --max-model-len 40000

python summary_cot.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 300 --max-tokens-sum 512 --max-model-len 40000