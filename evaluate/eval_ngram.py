import json
import argparse
import sys
import os
import copy
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from utils.novel_ngram import *


input_key = {
    "xsum": "document",
    "cnn_dm": "article",
    "ccsum": "article"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights",
    "ccsum": "summary"
}

def mean_score(scores):
    return sum(scores) / len(scores)

def extract_filename(json_path):
    # Get the basename (filename with extension)
    basename = os.path.basename(json_path)

    # Split the basename and extension
    filename, _ = os.path.splitext(basename)
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the prediction file (.json)")
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'ccsum'])
    parser.add_argument("--max_ngram_size", default=3, type=int, help="How many novel n-grams to check")

    args = parser.parse_args()

    data_path = args.data_path
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    
    ngram_results = defaultdict(list)
    annotated_samples = []
    for idx, sample in tqdm(enumerate(data)):
        document = sample[input_key[args.dataset]]
        gold_summary = sample[output_key[args.dataset]]
        pred_summary = sample['generated_summary']
        annotated_sample = copy.deepcopy(sample)

        ngrams = n_gram_percent(pred_summary, document, args.max_ngram_size)
        for metric, value in ngrams.items():
            ngram_results[metric].append(value)
        
        annotated_sample["novel_ngram"] = ngrams
        annotated_samples.append(annotated_sample)
    
    log_path = os.path.join("logs", extract_filename(args.data_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    file_path = os.path.basename(args.data_path)
    exp_name = os.path.splitext(file_path)[0]

    evaluation_metrics = {}
    evaluation_metrics["exp_name"] = exp_name
    for metric in sorted(ngram_results.keys()):
        scores = ngram_results[metric]
        evaluation_metrics[metric] = mean_score(scores)
    
    print(evaluation_metrics)
    # Check if average_metrics.json already exsits, if not exsit, us "w" mode otherwise use "a" mode
    save_path = os.path.join(log_path, "average_metrics.json")
    if not os.path.exists(log_path):
        with open(save_path, "w") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
    else:
        with open(save_path, "a") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
    
    with open(os.path.join(log_path, "ngram_metrics_per_sample.json"), "w") as fout:
        json.dump(annotated_samples, fout, indent=4)
    
