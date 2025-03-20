# Extract ContextCite attribution
#!/bin/bash

# Set the parameters
datasets=("ccsum" "xsum" "cnn_dm")
MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_SAMPLES=1000
NUM_SENTS=3

for dataset in "${datasets[@]}"; do
    echo "Starting attribution for dataset: $dataset"
    
    # Define the save path
    SAVE_PATH="results/${dataset}-llama3.1-8b-cc-1000.json"
    
    # Run the attribution script
    echo "python cc_attribution.py --dataset $dataset --num_samples $NUM_SAMPLES --num_sents $NUM_SENTS --save_path $SAVE_PATH --model_name $MODEL"
    
    python cc_attribution.py \
        --dataset "$dataset" \
        --num_samples "$NUM_SAMPLES" \
        --num_sents "$NUM_SENTS" \
        --save_path "$SAVE_PATH" \
        --model_name "$MODEL"

done