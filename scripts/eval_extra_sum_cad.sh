#!/usr/bin/env bash
#SBATCH -J cad+cfp-sum
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
DATASET=${2:-"cnn_dm"}
SCHEMA=${3:-"base"}

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
elif [ "$MODEL" = "vicuna-13b" ]; then
    MODEL_NAME="lmsys/vicuna-13b-v1.5"
elif [ "$MODEL" = "llama-13b" ]; then
    MODEL_NAME="huggyllama/llama-13b"
else
    echo "Invalid model option"
fi

set -x;
python summarisation_new_cad.py --dataset ${DATASET} \
                                --model_name ${MODEL_NAME} \
                                --num_samples 3000 \
                                --use_cad \
                                --alpha 0.5 \
                                --schema ${SCHEMA} \
                                --log_path results/${DATASET} \
                                --exp_name cad-${MODEL}-zero-shot-prompt-${SCHEMA} \
