#!/usr/bin/env bash

# Activate conda env
source ~/.bashrc
conda activate ~/env/mlkc

echo "Activated conda environment"

SCHEMA=("base" "attr" "instr" "opin" "instr+opin")
MODEL=("chat")  # TODO: use "base" to run experiments with Llama2-7b
DEMO=("none" "original")  # Demo mode

for p1 in "${DEMO[@]}"; do
    # Run the Python script with the current combination of parameters
    for p2 in "${SCHEMA[@]}"; do
        set -x;
        python3 llama_abstention.py --model_name meta-llama/Llama-2-7b-chat-hf \
                                    --schema ${p2} \
                                    --demo_mode ${p1} \
                                    --ans_mode mean \
                                    --log_path results/realtime_qa \
                                    --exp_name llama2-7b-chat-demo-${p1}-prompt-${p2}
    done
done