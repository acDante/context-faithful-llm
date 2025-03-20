# Generate summary with ContextCite attribution
#!/bin/bash

# Exit on error
set -e

# Check whether a dataset is provided as command line argument
if [ $# -eq 0 ]; then
    # Default dataset if none provided
    DATASET="ccsum"
    echo "No dataset specified, using default: $DATASET"
else
    # Use the provided dataset name
    DATASET="$1"
    echo "Running experiments on dataset: $DATASET"
fi

# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_SHORT_NAME="mistral-7b"

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT_NAME="llama3.1-8b"

# Check whether experiment directory and attribution file exist
ATTR_DATA_PATH="/mnt/ceph_rbd/project/context-faithful-llm/attribution/results/${DATASET}-${MODEL_SHORT_NAME}-cc-1000.json"
LOG_PATH="results/summary/$DATASET/${MODEL_SHORT_NAME}"

# Validate that attribution data path exists
if [ ! -f "$ATTR_DATA_PATH" ]; then
  echo "Error: Attribution data file $ATTR_DATA_PATH not found"
  exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_PATH"

schema_methods=("base" "base+impt" "impt_only" "impt+cad") 

for schema in "${schema_methods[@]}"; do
    if [ "$schema" = "impt+cad" ]; then
        python generate_summary.py --model_name ${MODEL_NAME} \
                                   --dataset $DATASET \
                                   --attr_data_path $ATTR_DATA_PATH \
                                   --num_samples 1000 \
                                   --log_path $LOG_PATH \
                                   --exp_name $DATASET-${MODEL_SHORT_NAME}-cc-$schema \
                                   --schema "base+impt" \
                                   --use_cad --alpha 0.5
    else
        python generate_summary.py --model_name ${MODEL_NAME} \
                                   --dataset $DATASET \
                                   --attr_data_path $ATTR_DATA_PATH \
                                   --num_samples 1000 \
                                   --log_path $LOG_PATH \
                                   --exp_name $DATASET-${MODEL_SHORT_NAME}-cc-$schema \
                                   --schema "$schema"
    fi
done