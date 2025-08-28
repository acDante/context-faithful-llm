#!/bin/bash

# python generate_attr_summary.py --model_name gpt-4o-mini --dataset xsum --num_samples 1000 --num_sents 3 --log_path results/summary/xsum/gpt-4o-mini --exp_name xsum-gpt-4o-mini-base --max_new_tokens 1024 --method base
# python generate_attr_summary.py --model_name gpt-4o-mini --dataset xsum --num_samples 1000 --num_sents 3 --log_path results/summary/xsum/gpt-4o-mini --exp_name xsum-gpt-4o-mini-gen_attr --max_new_tokens 1024 --method gen_attr

# python generate_attr_summary.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset xsum --num_samples 1000 --num_sents 3 --log_path results/summary/xsum/llama3.1-8b --exp_name xsum-llama3.1-8b-gen_attr --max_new_tokens 1024 --method gen_attr
# python generate_attr_summary.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset xsum --num_samples 20 --num_sents 3 --log_path results/summary/xsum/mistral-7b --exp_name xsum-mistral-7b-gen_attr --max_new_tokens 1024 --method gen_attr

# Run experiments with Qwen3 models
#!/bin/bash

# Define arrays for models and methods
#!/bin/bash

DATASET=${1:-"xsum"}

# Define arrays for models with [full_name,short_name] pairs
declare -A model_mapping
# model_mapping["Qwen/Qwen3-1.7B"]="qwen3-1.7b"
model_mapping["Qwen/Qwen3-4B"]="qwen3-4b"
model_mapping["Qwen/Qwen3-8B"]="qwen3-8b"
model_mapping["Qwen/Qwen3-14B"]="qwen3-14b"
model_mapping["Qwen/Qwen3-32B"]="qwen3-32b"

# Methods
methods=("base" "gen_attr")

# Loop through each model and method combination
for full_model_name in "${!model_mapping[@]}"; do
  # Get the short model name for paths and experiment naming
  short_model_name="${model_mapping[$full_model_name]}"
  
  for method in "${methods[@]}"; do
    echo "Running experiment with model: $full_model_name (short name: $short_model_name)"
    echo "Dataset: $DATASET"
    echo "Method: $method"

    # Set max_new_tokens based on method
    if [ "$method" = "base" ]; then
      max_tokens=128
    else
      max_tokens=1024
    fi
    
    # Create results directory if it doesn't exist
    mkdir -p "results/summary/$DATASET/$short_model_name"
    
    # Set experiment name using short model name
    exp_name="$DATASET-$short_model_name-$method-1000"
    
    # Run the command with appropriate parameters
    set -x;
    python generate_attr_summary.py \
      --model_name "$full_model_name" \
      --dataset $DATASET \
      --num_samples 1000 \
      --num_sents 3 \
      --log_path "results/summary/$DATASET/$short_model_name" \
      --exp_name "$exp_name" \
      --max_new_tokens $max_tokens \
      --method "$method"
    
    echo "Completed experiment: $exp_name"
    echo "-------------------------"
  done
done

echo "All experiments completed successfully!"