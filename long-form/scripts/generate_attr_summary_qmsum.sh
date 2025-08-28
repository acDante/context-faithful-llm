#!/bin/bash
# Experiments with generative attribution on QMSum

# set -x;
# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_qmsum_validation_272_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_qmsum_validation_272_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_qmsum_validation_272_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_qmsum_validation_272_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt_prefix

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_qmsum_validation_272_gen_attr_sentence_num70.json --attr_type sent70 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_qmsum_validation_272_gen_attr_sentence_num70.json --attr_type sent70 --method base+impt_prefix

# Experiments with generative attribution on QMSum
# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-8b_qmsum_validation_272_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-8b_qmsum_validation_272_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_qmsum_validation_272_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_qmsum_validation_272_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

# Experiments with ContextCite attribution on QMSum
# DATA_PATH="/mnt/ceph_rbd/project/context-faithful-llm/attribution/results"

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path ${DATA_PATH}/qmsum-llama3.1-8b-cc-30-sents-full-ablation-64_preds.json --attr_type cc_sent30 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path ${DATA_PATH}/qmsum-llama3.1-8b-cc-30-sents-full-ablation-64_preds.json --attr_type cc_sent30 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path ${DATA_PATH}/qmsum-qwen3-8b-cc-30-sents-full-ablation-64_preds.json --attr_type cc_sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path ${DATA_PATH}/qmsum-qwen3-8b-cc-30-sents-full-ablation-64_preds.json --attr_type cc_sent30 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_qmsum_validation_272_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_qmsum_validation_272_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_qmsum_validation_272_gen_attr_sentence_num70.json --attr_type sent70 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset qmsum --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_qmsum_validation_272_gen_attr_sentence_num70.json --attr_type sent70 --method base+impt_prefix

# 4. Ablation study with different number fo attributed sentences
DATA_PATH="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/ablation"

# Array of attributed sentence numbers to test
attr_nums=(5 10 20 30 40 50)

# Loop through each attribution number
for num in "${attr_nums[@]}"; do
    echo "Running experiment with ${num} attributed sentences..."
    
    set -x;
    python eval_scrolls.py \
        --model Qwen/Qwen3-8B \
        --dataset qmsum \
        --save-path results/ablation \
        --split train \
        --max-samples 100 \
        --max-tokens 512 \
        --gpu-memory-utilization 0.9 \
        --attr_data_path "${DATA_PATH}/qwen3-8b_qmsum_train_100_gen_attr_sentence_num${num}.json" \
        --attr_type "sent${num}" \
        --method base+impt
    
    echo "Completed experiment with ${num} attributed sentences."
done

# Run the baseline
python eval_scrolls.py --model Qwen/Qwen3-8B --dataset qmsum --save_path results/ablation --split train --max-samples 100 --max-tokens 512 --gpu-memory-utilization 0.9 --method base