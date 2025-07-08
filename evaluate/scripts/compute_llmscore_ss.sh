# Script for evaluating faithfulenss score for results on SummScreenFD
#!/bin/bash

data_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/summary"
attr_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/attribution"

# Run evaluation on summary files
# for json_file in ${data_path}/*.json; do
#     set -x;
#     python eval_with_llm.py --data_path "$json_file" \
#                             --dataset summscreen \
#                             --model_name Qwen/Qwen3-32B \
#                             --metrics consistency
# done

# Run on specific attribution files
# for num in 3 5 7; do
#     json_file="${data_path}/qwen3-32b_summscreen_validation_338_attr-fact-${num}_base+impt.json"
#     set -x;
#     python eval_with_llm.py --data_path "$json_file" \
#                             --dataset summscreen \
#                             --model_name Qwen/Qwen3-32B \
#                             --metrics consistency
# done

# Run on specific attribution files
for num in 3 5 7; do
    json_file="${attr_path}/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num${num}.json"
    set -x;
    python eval_with_llm.py --data_path "$json_file" \
                            --dataset summscreen \
                            --model_name Qwen/Qwen3-32B \
                            --metrics consistency
done