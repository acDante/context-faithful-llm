#!/usr/bin/env bash
#SBATCH -J cad+cfp-nq
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

SCHEMA=${1:-"base"}
ALPHA=${2:-"0.5"}

set -x;
python3 knowledge_conflict.py --model_name meta-llama/Llama-2-7b-chat-hf \
                                --orig_path ./datasets/llama_nq_data/nq_llama_orig.json \
                                --counter_path ./datasets/llama_nq_data/nq_llama_conflicted.json \
                                --schema ${SCHEMA} \
                                --demo_mode none \
                                --use_cad \
                                --alpha ${ALPHA} \
                                --log_path results/cad \
                                --exp_name llama2-7b-no-demo-prompt-${SCHEMA}-alpha-${ALPHA} \
