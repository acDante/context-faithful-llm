# Extract ContextCite attribution for long context experiments
#!/bin/bash

# python cc_attribution.py --dataset gov_report --model_name meta-llama/Llama-3.1-8B-Instruct --num_samples 300 --num_sents 30 --num_ablations 256 --save_path results/gov_report-llama3.1-8b-cc-30-sents-300-ablation-256_preds.json
# python cc_attribution.py --dataset gov_report --model_name Qwen/Qwen3-8B --num_samples 300 --num_sents 30 --num_ablations 256 --save_path results/gov_report-qwen3-8b-cc-30-sents-300-ablation-256_preds.json
python cc_attribution.py --dataset qmsum --model_name Qwen/Qwen3-8B --num_samples 300 --num_sents 30 --num_ablations 128 --save_path results/qmsum-qwen3-8b-cc-30-sents-full-ablation-128_preds.json
python cc_attribution.py --dataset qmsum --model_name meta-llama/Llama-3.1-8B-Instruct --num_samples 300 --num_sents 30 --num_ablations 64 --save_path results/qmsum-llama3.1-8b-cc-30-sents-full-ablation-64_preds.json