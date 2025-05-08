#!/bin/bash
# Generate summaries using attribution-guided generation

DATASET=${1:-"xsum"}
MODEL=${2:-"llama3.1-8b"}
ATTRIBUTION="attention"
NUM_SAMPLES=100

if [ "$MODEL" = "llama3.1-8b" ]; then
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
elif [ "$MODEL" = "mistral-7b" ]; then
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
else
    echo "Invalid model option"
fi

echo "Running experiments on dataset $DATASET with model $MODEL"

# Run prompting-based method
schema_methods=("base+impt" "impt_only" "base+impt_prefix")

for schema in "${schema_methods[@]}"; do
    set -x;
    python generate_summary_new.py --model_name ${MODEL_NAME} \
                                    --dataset ${DATASET} \
                                    --attr_data_path ../attribution/results/${DATASET}-${MODEL}-${ATTRIBUTION}-mean-${NUM_SAMPLES}.json \
                                    --num_samples ${NUM_SAMPLES} \
                                    --log_path results/summary/${DATASET}/${MODEL} \
                                    --exp_name ${DATASET}-${MODEL}-${ATTRIBUTION}-${schema}-${NUM_SAMPLES} \
                                    --schema ${schema} \
                                    --max_new_tokens 128 \

done

# Run DoLA methods
python generate_summary_new.py --model_name ${MODEL_NAME} --dataset ${DATASET} --num_samples ${NUM_SAMPLES} --log_path results/summary/${DATASET}/${MODEL} --exp_name ${DATASET}-${MODEL}-base-dola-low-${NUM_SAMPLES} --schema base --method dola --dola_config low --max_new_tokens 128 --attr_data_path ../attribution/results/${DATASET}-${MODEL}-${ATTRIBUTION}-mean-${NUM_SAMPLES}.json

python generate_summary_new.py --model_name ${MODEL_NAME} --dataset ${DATASET} --num_samples ${NUM_SAMPLES} --log_path results/summary/${DATASET}/${MODEL} --exp_name ${DATASET}-${MODEL}-base-dola-high-${NUM_SAMPLES} --schema base --method dola --dola_config high --max_new_tokens 128 --attr_data_path ../attribution/results/${DATASET}-${MODEL}-${ATTRIBUTION}-mean-${NUM_SAMPLES}.json

# Run baseline CAD method
python generate_summary_new.py --model_name ${MODEL_NAME} --dataset ${DATASET} --num_samples ${NUM_SAMPLES} --log_path results/summary/${DATASET}/${MODEL} --exp_name ${DATASET}-${MODEL}-base-cad-${NUM_SAMPLES} --schema base --method cad --alpha 0.5 --max_new_tokens 128 --attr_data_path ../attribution/results/${DATASET}-${MODEL}-${ATTRIBUTION}-mean-${NUM_SAMPLES}.json


# Run attribution-guided CAD approach
cad_methods=("base+impt" "mask_impt")

for schema in "${cad_methods[@]}"; do
    set -x;
    python generate_summary_new.py --model_name ${MODEL_NAME} \
                                    --dataset ${DATASET} \
                                    --attr_data_path ../attribution/results/${DATASET}-${MODEL}-${ATTRIBUTION}-mean-${NUM_SAMPLES}.json \
                                    --num_samples ${NUM_SAMPLES} \
                                    --log_path results/summary/${DATASET}/${MODEL} \
                                    --exp_name ${DATASET}-${MODEL}-${ATTRIBUTION}-${schema}-cad-${NUM_SAMPLES} \
                                    --schema ${schema} \
                                    --method cad \
                                    --alpha 0.5 \
                                    --max_new_tokens 128 \

done