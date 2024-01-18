#!/usr/bin/env bash
#SBATCH -J cfllm-nq
#SBATCH -A NLP-CDT-SL4-GPU
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

MODEL=${1:-"base"}
SCHEMA=${2:-"base"}

if [ "$MODEL" = "base" ]; then
    MODEL_NAME="meta-llama/Llama-2-7b-hf"
elif [ "$MODEL" = "chat" ]; then
    MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
else
    echo "Invalid model option"
fi

set -x;
python knowledge_conflict.py --model_name ${MODEL_NAME} \
                             --schema ${SCHEMA} \
                             --demo_mode none \
                             --num_demos 1 \
                             --exp_name llama2-7b-${MODEL}-no-demo-${SCHEMA} \

# schema = ['base', 'attr', 'instr', 'opin', 'instr+opin']
# demo_mode = ['none', 'original', 'counter']