"""Compute evaluation metrics given the predictions on summarisation data"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import copy
import nltk


input_key = {
    "xsum": "document",
    "cnn_dm": "article"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights"
}

def mean_score(scores):
    return sum(scores) / len(scores)

def postprocess_text(preds, labels):
    """ Postprocessing predictions and references for computing rouge-L scores
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dm", type=str, choices=['cnn_dm', 'xsum'])   
    parser.add_argument("--pred_file", type=str, help="Path to the prediction file for evaluation (.json)")
    parser.add_argument("--log_path", type=str, help="Store the evaluation metrics to log file (.log)")

    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    access_token = "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"
    download_path = "/home/hpcdu1/experiments/huggingface-hub"

    # Load references and predictions
    with open(args.pred_file, 'r') as fin:
        pred_data = json.load(fin)

    docs = [data[input_key[args.dataset]] for data in pred_data]
    references = [data[output_key[args.dataset]] for data in pred_data]
    predictions = [data['prediction'] for data in pred_data]

    # Compute the ROUGE scores
    # Note: need to post-process the text before computing Rouge-L scores
    print("Computing Rouge scores..")
    rouge = evaluate.load("rouge")
    processed_preds, processed_refs = postprocess_text(predictions, references)
    rouge_scores = rouge.compute(predictions=processed_preds, 
                                 references=processed_refs,
                                 use_aggregator=False)
    
    # Compute BERT score
    print("Computing BERT scores..")
    bert_score = evaluate.load('bertscore')
    bert_score_res = bert_score.compute(predictions=predictions, 
                                        references=references, 
                                        model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    # Load FactKB model
    factkb_tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True,
                                                     cache_dir=download_path)
    factkb_model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2, 
                                                                      device_map="auto", cache_dir=download_path)
   # Evaluate FactKB score
    print("Computing FactKB scores..")
    factkb_scores = []

    # Add evaluation metric for each instance
    annotated_examples = []
    for data in tqdm(pred_data):
        doc = data['article']
        ref = data['highlights']
        pred = data['prediction']
        annotated_example = copy.deepcopy(data)

        # Compute ROUGE score for each instance
        rouge = evaluate.load("rouge")
        processed_pred, processed_ref = postprocess_text([pred], [ref])
        rouge_score = rouge.compute(predictions=processed_pred, 
                                     references=processed_ref,
                                     use_aggregator=True)

        for rouge_type in ['rouge1', 'rouge2', 'rougeLsum']:
            annotated_example[rouge_type] = rouge_score[rouge_type]

        # Compute factKB score for each instance
        factkb_input = [[pred, doc]]
        factkb_tokens = factkb_tokenizer(factkb_input, return_tensors="pt", 
                                         padding="max_length", truncation=True).to(factkb_model.device)
        factkb_logits = factkb_model(**factkb_tokens).logits
        factkb_res = torch.softmax(factkb_logits, dim=1)
        factkb_score = float(factkb_res[0][1])
        factkb_scores.append(factkb_score)

        annotated_example['factkb'] = factkb_score
        annotated_examples.append(annotated_example)

    metrics = {
        'rouge1': mean_score(rouge_scores['rouge1']),
        'rouge2': mean_score(rouge_scores['rouge2']),
        'rougeLsum': mean_score(rouge_scores['rougeLsum']),
        'bertscore_p': mean_score(bert_score_res['precision']),
        'bertscore_r': mean_score(bert_score_res['recall']),
        'bertscore_f1': mean_score(bert_score_res['f1']),
        "factKB": mean_score(factkb_scores)
    }

    # Print the evaluation metrics
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    if args.log_path:
        log_path = Path(args.log_path)
        parts = args.pred_file.split('/')
        exp_name = parts[-1].split('.')[0]
        save_path = log_path / f"{exp_name}.log"
        print(f"Saving evaluation metrics to log file: {save_path}")
        with open(save_path, "w") as fout:
            for metric_name, value in metrics.items():
                fout.write(f"{metric_name}: {value:.4f}")
                fout.write("\n")
        
        save_path = log_path / f"annotated-{exp_name}.json"
        with open(save_path, 'w') as fout:
            json.dump(annotated_examples, fout, indent=4)

if __name__ == '__main__':
    main()