#!/bin/bash

# for type in fact sentence; do
#     for num in 3 5 7; do
#         set -x;
#         python generate_attr_summary.py --model Qwen/Qwen3-32B \
#                                         --dataset summscreen \
#                                         --max-samples 1000 \
#                                         --num_sents ${num} \
#                                         --attr_type ${type}
#     done
# done

for type in fact sentence; do
    for num in 3 5 7; do
        attr_path="results/attribution/qwen3-32b_summscreen_validation_338_gen_attr_${type}_num${num}.json"
        set -x;
        python eval_scrolls.py --model Qwen/Qwen3-32B \
                               --dataset summscreen \
                               --max-samples 1000 \
                               --attr_data_path "$attr_path" \
                               --attr_type ${type}-${num}\
                               --method base+impt
    done
done