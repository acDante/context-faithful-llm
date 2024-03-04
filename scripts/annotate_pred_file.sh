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

set -x;
python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-7b-chat-zero-shot-prompt-opin_preds.json --log_path results/eval
python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-7b-chat-zero-shot-prompt-opin_preds.json --log_path results/eval

python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-7b-zero-shot-prompt-base_preds.json --log_path results/eval
python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-7b-zero-shot-prompt-base_preds.json --log_path results/eval

python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-13b-zero-shot-prompt-instr+opin_preds.json --log_path results/eval
python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-13b-zero-shot-prompt-instr+opin_preds.json --log_path results/eval

python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-13b-chat-zero-shot-prompt-instr+opin_preds.json --log_path results/eval
python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-13b-chat-zero-shot-prompt-instr+opin_preds.json --log_path results/eval

# python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-13b-chat-zero-shot-prompt-instr_preds.json --log_path results/eval
# python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-13b-chat-zero-shot-prompt-instr_preds.json --log_path results/eval

# python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-13b-chat-zero-shot-prompt-opin_preds.json --log_path results/eval
# python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-13b-chat-zero-shot-prompt-opin_preds.json --log_path results/eval

# python eval_sum.py --dataset cnn_dm --pred_file results/xsum/llama2-13b-chat-zero-shot-prompt-attr_preds.json --log_path results/eval
# python eval_sum.py --dataset cnn_dm --pred_file results/xsum/cad-llama2-13b-chat-zero-shot-prompt-attr_preds.json --log_path results/eval