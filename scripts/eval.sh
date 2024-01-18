#!/usr/bin/env bash

# Activate conda env
source ~/.bashrc
conda activate ~/env/mlkc

echo "Activated conda environment"

SCHEMA=("base" "attr" "instr" "opin" "instr+opin")
MODEL=("chat")  # TODO: use "base" to run experiments with Llama2-7b
DEMO=("none" "original" "counter")  # Demo mode

for p1 in "${DEMO[@]}"; do
    # Run the Python script with the current combination of parameters
    for p2 in "${SCHEMA[@]}"; do
        set -x;
        python3 knowledge_conflict.py --model_name meta-llama/Llama-2-7b-chat-hf \
                                    --orig_path ./datasets/llama_nq_data/nq_llama_orig.json \
                                    --counter_path ./datasets/llama_nq_data/nq_llama_conflicted.json \
                                    --schema ${p2} \
                                    --demo_mode ${p1} \
                                    --num_demos 16 \
                                    --log_path results/llama_nq_data \
                                    --exp_name llama2-7b-chat-demo-${p1}-prompt-${p2}
    done
done


# # Zero-shot experiments
# python3 knowledge_conflict.py --model_name ${MODEL_NAME} \
#                             --orig_path ./datasets/nq_llama2/orig_dev_filtered.json \
#                             --counter_path ./datasets/nq_llama2/conflict_dev_filtered.json \
#                             --schema ${p1} \
#                             --demo_mode none \
#                             --num_demos 1 \
#                             --log_path results/filtered_nq_llama2 \
#                             --exp_name llama2-7b-${p2}-no-demo-${p1}

# In-context learning experiments
# python3 knowledge_conflict.py --model_name ${MODEL_NAME} \
#                             --orig_path ./datasets/nq_llama2/orig_dev_filtered.json \
#                             --counter_path ./datasets/nq_llama2/conflict_dev_filtered.json \
#                             --schema ${p1} \
#                             --demo_mode counter \
#                             --num_demos 16 \
#                             --log_path results/filtered_nq_llama2 \
#                             --exp_name llama2-7b-${p2}-demo-counter-${p1}

# python3 knowledge_conflict.py --model_name ${MODEL_NAME} \
#                             --orig_path ./datasets/nq_llama2/orig_dev_filtered.json \
#                             --counter_path ./datasets/nq_llama2/conflict_dev_filtered.json \
#                             --schema ${p1} \
#                             --demo_mode original \
#                             --num_demos 16 \
#                             --log_path results/filtered_nq_llama2 \
#                             --exp_name llama2-7b-${p2}-demo-original-${p1}

# Run experiments with Llama-2 models
# for p2 in "${MODEL[@]}"; do
#     if [ "$p2" = "base" ]; then
#         MODEL_NAME="meta-llama/Llama-2-7b-hf"
#     elif [ "$p2" = "chat" ]; then
#         MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
#     else
#         echo "Invalid model option"
#     fi