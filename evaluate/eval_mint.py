# Compute the MINT score for the prediction

import json
import argparse
from tqdm import tqdm
import nltk
import os 
import copy
from utils.mint import *

input_key = {
    "xsum": "document",
    "cnn_dm": "article",
    "ccsum": "article",
    "summscreen": "input",
    "qmsum": "input",
    "gov_report": "input"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights",
    "ccsum": "summary",
    "summscreen": "output",
    "qmsum": "output",
    "gov_report": "output"
}

def extract_filename(json_path):
    # Get the basename (filename with extension)
    basename = os.path.basename(json_path)

    # Split the basename and extension
    filename, _ = os.path.splitext(basename)
    return filename

def mean_score(scores):
    return sum(scores) / len(scores)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the prediction file (.json)")
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum', 'summscreen', 'qmsum', 'gov_report'])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for MINT evaluation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    data_path  = args.data_path
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    
    print(f"Loaded {len(data)} samples at: {data_path}")
    
    documents = []
    pred_summaries = []

    for sample in data:
        document = sample[input_key[args.dataset]]
        pred_summary = sample['generated_summary']
        documents.append(document)
        pred_summaries.append(pred_summary)
    
    scores = evaluate_mint(pred_summaries, documents, batch_size=args.batch_size)
    mint_scores = [score['mint'] for score in scores]

    annotated_samples = []
    for sample, mint_score in zip(data, mint_scores):
        annotated_sample = copy.deepcopy(sample)
        annotated_sample["mint_score"] = float(mint_score)
        annotated_samples.append(annotated_sample)
    
    log_path = os.path.join("logs", extract_filename(args.data_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    evaluation_metrics = {}
    file_path = os.path.basename(args.data_path)
    exp_name = os.path.splitext(file_path)[0]
    evaluation_metrics["exp_name"] = exp_name
    evaluation_metrics["mint_score"] = mean_score(mint_scores)

    print(f"Average MINT score: {evaluation_metrics['mint_score']:.4f}")
    save_path = os.path.join(log_path, "average_metrics.json")
    if not os.path.exists(log_path):
        with open(save_path, "w") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
            fout.write('\n')
    else:
        with open(save_path, "a") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
            fout.write('\n')
    
    with open(os.path.join(log_path, "mint_metrics_per_sample.json"), "w") as fout:
        json.dump(annotated_samples, fout, indent=4)