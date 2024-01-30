#!/usr/bin/env bash
#SBATCH -J cad+cfp-nq
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
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

set -x;
python3 summarisation.py --dataset cnn_dm \
                         --model_name mistralai/Mistral-7B-Instruct-v0.1 \
                         --schema ${SCHEMA} \
                         --log_path results/cnndm \
                         --exp_name mistral-7b-instr-zero-shot-prompt-${SCHEMA} \