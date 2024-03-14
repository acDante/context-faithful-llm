# Evaluate the quality of extracted sentences using attribution method

import torch
import torch.nn as nn
import datasets
import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import nltk

def eval_rouge_scores(preds, labels):
    processed_preds, processed_labels = postprocess_text(preds, labels)
    rouge_scores = rouge.compute(predictions=processed_preds,
                                 references=processed_labels)
    
    metrics = {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeLsum': rouge_scores['rougeLsum']
    }

    return metrics

def print_metrics(metrics):
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

def postprocess_text(preds, labels):
    """ Postprocessing predictions and references for computing rouge L scores
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the JSON file containing the extracted important sentences")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    data_path = args.data_path
    with open(data_path, "r") as fin:
        data = json.load(fin)
    
    abs_references = []
    extra_references = []
    extra_preds = []

    for idx, sample in tqdm(enumerate(data)):
        gold_summary = " ".join(sample['tgt'])  # gold abstractive summary
        ref_extra_summary = " ".join(sample['extractive_summary'])  # annotated extractive summary
        important_sents = " ".join(sample['important_sents'])  # important sentences extracted by attribution

        abs_references.append(gold_summary)
        extra_references.append(ref_extra_summary)
        extra_preds.append(important_sents)

    # Compute the ROUGE scores by comparing the abstractive summary and two extractive summaries
    rouge = evaluate.load('rouge')
    ref_metrics = eval_rouge_scores(extra_references, abs_references)
    metrics = eval_rouge_scores(extra_preds, abs_references)

    # Print the ROUGE scores
    print("ROUGE scores for the reference extractive summary:")
    print_metrics(ref_metrics)

    print("ROUGE scores for the predicted extractive summary:")
    print_metrics(metrics)
