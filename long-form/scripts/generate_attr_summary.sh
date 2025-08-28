#!/bin/bash

# 1. Experiments with generative attribution on GovReport
# set -x;
# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --method base

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path results/attribution/qwen3-8b_gov_report_validation_300_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path results/attribution/qwen3-8b_gov_report_validation_300_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

# 2. Experiments with Contextcite attribution on GovReport
# DATA_PATH="/mnt/ceph_rbd/project/context-faithful-llm/attribution/results"

# set -x;
# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path ${DATA_PATH}/gov_report-llama3.1-8b-cc-30-sents-300_preds.json --attr_type cc_sent30 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path ${DATA_PATH}/gov_report-llama3.1-8b-cc-30-sents-300_preds.json --attr_type cc_sent30 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path ${DATA_PATH}/gov_report-qwen3-8b-cc-30-sents-300_preds.json --attr_type cc_sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path ${DATA_PATH}/gov_report-qwen3-8b-cc-30-sents-300_preds.json --attr_type cc_sent30 --method base+impt_prefix

# set -x;
# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path ${DATA_PATH}/gov_report-llama3.1-8b-cc-30-sents-300-ablation-128_preds.json --attr_type cc_128_sent30 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gov_report --max-samples 300 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --attr_data_path ${DATA_PATH}/gov_report-llama3.1-8b-cc-30-sents-300-ablation-128_preds.json --attr_type cc_128_sent30 --method base+impt_prefix

# 3. Add SumCoT baseline
# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gov_report --max-samples 300 --max-tokens 5000 --max-model-len 80000 --gpu-memory-utilization 0.9 --method sum_cot

# 4. Ablation study with different number of attributed sentences
DATA_PATH="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/ablation"

# Array of attributed sentence numbers to test
attr_nums=(5 10 20 30 40 50)

# Loop through each attribution number
for num in "${attr_nums[@]}"; do
    echo "Running experiment with ${num} attributed sentences..."
    
    set -x;
    python eval_scrolls.py \
        --model Qwen/Qwen3-8B \
        --dataset gov_report \
        --save-path results/ablation \
        --split train \
        --max-samples 100 \
        --max-tokens 1024 \
        --max-model-len 80000 \
        --gpu-memory-utilization 0.9 \
        --attr_data_path "${DATA_PATH}/qwen3-8b_gov_report_train_100_gen_attr_sentence_num${num}.json" \
        --attr_type "sent${num}" \
        --method base+impt
    
    echo "Completed experiment with ${num} attributed sentences."
done

# Run the baseline
python eval_scrolls.py --model Qwen/Qwen3-8B --dataset gov_report --save_path results/ablation --split train --max-samples 100 --max-tokens 1024 --max-model-len 80000 --gpu-memory-utilization 0.9 --method base