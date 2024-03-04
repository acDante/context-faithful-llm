#!/usr/bin/env bash
#SBATCH -J extra_sum
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --partition=ampere
#SBATCH -o /home/%u/slogs/slurm-%A.out
#SBATCH -e /home/%u/slogs/slurm-%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2334664@ed.ac.uk


# Activate conda env
source ~/.bashrc
conda activate ~/env/mlkc

echo "Activated conda environment"

MODEL=${1:-"llama2-7b"}
SCHEMA=${2:-"base"}

if [ "$MODEL" = "llama2-7b" ]; then
    MODEL_NAME="meta-llama/Llama-2-7b-hf"  # Use non-chat model
elif [ "$MODEL" = "llama2-7b-chat" ]; then
    MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
elif [ "$MODEL" = "llama2-13b" ]; then
    MODEL_NAME="meta-llama/Llama-2-13b-hf"
elif [ "$MODEL" = "llama2-13b-chat" ]; then
    MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
elif [ "$MODEL" = "mistral-7b" ]; then
    MODEL_NAME="mistralai/Mistral-7B-v0.1"
elif [ "$MODEL" = "mistral-7b-instr" ]; then
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
elif [ "$MODEL" = "llama-13b" ]; then
    MODEL_NAME="huggyllama/llama-13b"
else
    echo "Invalid model option"
fi

set -x;
python extractive_sum.py --model_name ${MODEL_NAME} \
                        --num_samples 3000 \
                        --schema ${SCHEMA} \
                        --log_path results/extra_cnn\
                        --exp_name ${MODEL}-zero-shot-prompt-${SCHEMA} \
