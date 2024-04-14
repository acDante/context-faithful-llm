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


def mean_score(scores):
    return sum(scores) / len(scores)

def eval_rouge_scores(preds, labels):
    rouge = evaluate.load('rouge')
    processed_preds, processed_labels = postprocess_text(preds, labels)
    rouge_scores = rouge.compute(predictions=processed_preds,
                                 references=processed_labels)
    metrics = {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeLsum': rouge_scores['rougeLsum']
    }

    return metrics

def eval_bert_scores(preds, labels):
    bert_score = evaluate.load('bertscore')
    bert_score_res = bert_score.compute(predictions=preds, 
                                        references=labels, 
                                        model_type="microsoft/deberta-xlarge-mnli", lang="en")
    metrics = {
        'bertscore_p': mean_score(bert_score_res['precision']),
        'bertscore_r': mean_score(bert_score_res['recall']),
        'bertscore_f1': mean_score(bert_score_res['f1']),
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
    ref_rouge_metrics = eval_rouge_scores(extra_references, abs_references)
    rouge_metrics = eval_rouge_scores(extra_preds, abs_references)

    # Print the ROUGE scores
    print("ROUGE scores for the reference extractive summary:")
    print_metrics(ref_rouge_metrics)

    print("ROUGE scores for the predicted extractive summary:")
    print_metrics(rouge_metrics)

    # Print the BERT scores
    ref_bert_score_metrics = eval_bert_scores(extra_references, abs_references)
    bert_score_metrics = eval_bert_scores(extra_preds, abs_references)

    print("BERT scores for the reference extractive summary:")
    print_metrics(ref_bert_score_metrics)

    print("BERT scores for the predicted extractive summary:")
    print_metrics(bert_score_metrics)

    # Compute the average sentence-level precision
    # i.e. the percentage of sentences in the predicted summary that are present in the gold summary
    precision = []
    recall = []
    for idx, sample in tqdm(enumerate(data)):
        reference = sample['extractive_summary']
        pred = sample['important_sents']
        count = 0
        for sent in pred:
            if sent in reference:
                count += 1
        precision.append(count / len(pred))
        recall.append(count / len(reference))
    
    print(f"Average sentence-level precision: {mean_score(precision):.4f}")
    print(f"Average sentence-level recall: {mean_score(recall):.4f}")
            