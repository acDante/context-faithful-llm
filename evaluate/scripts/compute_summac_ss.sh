# Script for evaluating ROUGE-L, BERTScore, SummaC for results on SummScreenFD
#!/bin/bash

data_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/summary"
attr_path="/mnt/ceph_rbd/project/context-faithful-llm/long-form/results/attribution"

# Run evaluation on summary files
for json_file in ${data_path}/*.json; do
    set -x;
    python new_eval_basic.py --data_path "$json_file" \
                             --dataset summscreen

done

# Run on specific attribution files
for json_file in ${attr_path}/*.json; do
    set -x;
    python new_eval_basic.py --data_path "$json_file" \
                             --dataset summscreen

done

# for num in 3 5 7; do
#     json_file="${attr_path}/qwen3-32b_summscreen_validation_338_gen_attr_sentence_num${num}.json"
#     set -x;
#     python new_eval_basic.py --data_path "$json_file" \
#                             --dataset summscreen
                
# done