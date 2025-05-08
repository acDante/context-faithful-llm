# Generate summaries using attribution-guided generation
# Extract attribution using lead-3 and random baseline

DATASET=${1:-"xsum"}
MODEL=${2:-"llama3.1-8b"}
ATTRIBUTION=${3:-"lead3"}
NUM_SAMPLES=1000

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
                                    --attr_data_path ../attribution/results/${DATASET}-${ATTRIBUTION}-${NUM_SAMPLES}.json \
                                    --num_samples ${NUM_SAMPLES} \
                                    --log_path results/summary/${DATASET}/${MODEL} \
                                    --exp_name ${DATASET}-${MODEL}-${ATTRIBUTION}-${schema} \
                                    --schema ${schema} \
                                    --max_new_tokens 128 \

done

# Run attribution-guided CAD approach
cad_methods=("base+impt" "mask_impt")

for schema in "${cad_methods[@]}"; do
    set -x;
    python generate_summary_new.py --model_name ${MODEL_NAME} \
                                    --dataset ${DATASET} \
                                    --attr_data_path ../attribution/results/${DATASET}-${ATTRIBUTION}-${NUM_SAMPLES}.json \
                                    --num_samples ${NUM_SAMPLES} \
                                    --log_path results/summary/${DATASET}/${MODEL} \
                                    --exp_name ${DATASET}-${MODEL}-${ATTRIBUTION}-${schema}-cad \
                                    --schema ${schema} \
                                    --method cad \
                                    --alpha 0.5 \
                                    --max_new_tokens 128 \

done
