#!/bin/bash
# Experiments with generative attribution on SummScreenFD

# 3. Experiments with generative attribution on SummScreenFD
set -x;
python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_summscreen_validation_338_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_summscreen_validation_338_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

set -x;
python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-8b_summscreen_validation_338_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-8b_summscreen_validation_338_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

set -x;
# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num30.json --attr_type sent30 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num70.json --attr_type sent70 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-32B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num70.json --attr_type sent70 --method base+impt_prefix

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-8b_summscreen_validation_338_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt

# python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/qwen3-8b_summscreen_validation_338_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt_prefix

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --method base

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_summscreen_validation_338_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt

# python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path results/attribution/llama3.1-8b_summscreen_validation_338_gen_attr_sentence_num50.json --attr_type sent50 --method base+impt_prefix

# Evaluation with ContextCite attribution
python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path /mnt/ceph_rbd/project/context-faithful-llm/attribution/results/summscreen-llama3.1-8b-cc-50-sents-full_preds.json --attr_type sent50 --method base+impt

python eval_scrolls.py --model meta-llama/Llama-3.1-8B-Instruct --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path /mnt/ceph_rbd/project/context-faithful-llm/attribution/results/summscreen-llama3.1-8b-cc-50-sents-full_preds.json --attr_type sent50 --method base+impt_prefix

python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path /mnt/ceph_rbd/project/context-faithful-llm/attribution/results/summscreen-qwen3-8b-cc-50-sents-full_preds.json --attr_type sent50 --method base+impt

python eval_scrolls.py --model Qwen/Qwen3-8B --dataset summscreen --max-samples 500 --max-tokens 512 --max-model-len 34000 --attr_data_path /mnt/ceph_rbd/project/context-faithful-llm/attribution/results/summscreen-qwen3-8b-cc-50-sents-full_preds.json --attr_type sent50 --method base+impt_prefix