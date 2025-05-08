#!/bin/bash
# Script for generating summary on XSum/CCSum/CNN-DM using DoLA

# Activate conda env
source ~/.bashrc
conda activate llm

# # DoLA experiments with Mistral-7b
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset xsum --num_samples 1000 --log_path results/summary/xsum/mistral-7b --exp_name xsum-mistral-7b-base-dola-low --schema base --method dola --dola_config low --max_new_tokens 128
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset xsum --num_samples 1000 --log_path results/summary/xsum/mistral-7b --exp_name xsum-mistral-7b-base-dola-high --schema base --method dola --dola_config high --max_new_tokens 128
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/mistral-7b --exp_name ccsum-mistral-7b-base-dola-low --schema base --method dola --dola_config low --max_new_tokens 128
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/mistral-7b --exp_name ccsum-mistral-7b-base-dola-high --schema base --method dola --dola_config high --max_new_tokens 128
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset cnn_dm --num_samples 1000 --log_path results/summary/cnn_dm/mistral-7b --exp_name cnn_dm-mistral-7b-base-dola-low --schema base --method dola --dola_config low --max_new_tokens 128
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset cnn_dm --num_samples 1000 --log_path results/summary/cnn_dm/mistral-7b --exp_name cnn_dm-mistral-7b-base-dola-high --schema base --method dola --dola_config high --max_new_tokens 128

# # Baseline CAD experiments with Mistral-7b
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset xsum --num_samples 1000 --log_path results/summary/xsum/mistral-7b --exp_name xsum-mistral-7b-base-cad --schema base --method cad --alpha 0.5 --max_new_tokens 128
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/mistral-7b --exp_name ccsum-mistral-7b-base-cad --schema base --method cad --alpha 0.5 --max_new_tokens 128 --attr_data_path results/ccsum-mistral-7b-attention-1000.json

# # New CAD formulation
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset xsum --num_samples 1000 --log_path results/summary/xsum/mistral-7b --exp_name xsum-mistral-7b-mask_impt-cad --schema mask_impt --method cad --alpha 0.5 --max_new_tokens 128 --attr_data_path results/xsum-mistral-7b-attention-1000.json
# python generate_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/mistral-7b --exp_name ccsum-mistral-7b-mask_impt-cad --schema mask_impt --method cad --alpha 0.5 --max_new_tokens 128 --attr_data_path results/ccsum-mistral-7b-attention-1000.json  # This is not yet submitted

## DoLA expriments with Llama3.1-8b
set -x;
python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/llama3.1-8b --exp_name ccsum-llama3.1-8b-base-dola-low --schema base --method dola --dola_config low --max_new_tokens 128 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json
python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/llama3.1-8b --exp_name ccsum-llama3.1-8b-base-dola-high --schema base --method dola --dola_config high --max_new_tokens 128 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json

# python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset xsum --num_samples 1000 --log_path results/summary/xsum/llama3.1-8b --exp_name xsum-llama3.1-8b-base-dola-low --schema base --method dola --dola_config low --max_new_tokens 128 --attr_data_path ../attribution/results/xsum-llama3.1-8b-attention-mean-1000.json
# python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset xsum --num_samples 1000 --log_path results/summary/xsum/llama3.1-8b --exp_name xsum-llama3.1-8b-base-dola-high --schema base --method dola --dola_config high --max_new_tokens 128 --attr_data_path ../attribution/results/xsum-llama3.1-8b-attention-mean-1000.json

## CAD baseline with Llama3.1-8b
python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/llama3.1-8b --exp_name ccsum-llama3.1-8b-base-cad --schema base --method cad --alpha 0.5 --max_new_tokens 128 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json

# Lead3 baseline
python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/llama3.1-8b --exp_name ccsum-llama3.1-8b-base+lead3 --schema lead3 --max_new_tokens 128 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json

# Base+impt (prefix version)
python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset ccsum --num_samples 1000 --log_path results/summary/ccsum/llama3.1-8b --exp_name ccsum-llama3.1-8b-base+impt_prefix --schema base+impt_prefix --max_new_tokens 128 --attr_data_path ../attribution/results/ccsum-llama3.1-8b-attention-mean-1000.json

# python generate_summary_new.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset xsum --num_samples 1000 --log_path results/summary/xsum/llama3.1-8b/ --exp_name xsum-llama3.1-8b-base-cad --schema base --method cad --alpha 0.5 --max_new_tokens 128 --attr_data_path ../attribution/results/xsum-llama3.1-8b-attention-mean-1000.json