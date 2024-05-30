#!/bin/bash

# Activate conda env
source ~/.bashrc
conda activate inseq

attribution_methods=("attention")
# attribution_methods=("deeplift" "gradient_shap") 

for attribution_method in "${attribution_methods[@]}"
do
    python extract_sents.py --model_name google/flan-t5-base --num_samples 2500 --num_sents 3 --save_path results/extra_cnn-flan-t5-base-"$attribution_method"-new.json --attribution $attribution_method
done
