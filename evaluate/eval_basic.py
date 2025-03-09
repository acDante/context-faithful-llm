# Compute the ROUGE, BERT score and Summa-C score for the prediction

import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import nltk
import os

from summac.model_summac import SummaCZS, SummaCConv


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
    parser.add_argument("--data_path", type=str, help="Path to the prediction file (.json)")
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum'])
    parser.add_argument("--log_path", type=str, help="Path to save the evaluation results for each file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # Model for computing Summa-C scores
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda:0", start_file="default", agg="mean")

    data_path  = args.data_path
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    
    documents = []
    golds = []
    predictions = []

    # Collect Summa-C scores
    summac_scores = []

    for idx, sample in tqdm(enumerate(data)):
        document = sample[input_key[args.dataset]]
        gold_summary = sample[output_key[args.dataset]]
        pred_summary = sample['generated_summary']

        # Compute Summa-C score for each sample
        summac_score = model_conv.score([document], [pred_summary])
        summac_scores.append(summac_score["scores"][0])

        # Collect predictions and gold summaries
        documents.append(document)
        golds.append(gold_summary)
        predictions.append(pred_summary)
    
    log_path = Path(args.log_path)
    # with open(log_path, "a") as fout:
    #     fout.write(f"Currently evaluating: {args.data_path}\n")
    
    evaluation_metrics = {}
    file_path = os.path.basename(args.data_path)
    evaluation_metrics["exp_name"] = os.path.splitext(file_path)[0]

    # Compute ROUGE scores
    rouge = evaluate.load('rouge')
    rouge_metrics = eval_rouge_scores(predictions, golds)
    for metric_name, value in rouge_metrics.items():
        evaluation_metrics[metric_name] = value

    # with open(log_path, "a") as fout:
    #     fout.write("ROUGE scores for the predicted summary:\n")
    #     for metric_name, value in rouge_metrics.items():
    #         fout.write(f"{metric_name}: {value:.4f}\n")
    #         evaluation_metrics[metric_name] = value
    
    # Compute BERT scores
    bert_score_metrics = eval_bert_scores(predictions, golds)
    for metric_name, value in bert_score_metrics.items():
        evaluation_metrics[metric_name] = value

    # with open(log_path, "a") as fout:
    #     fout.write("BERT scores for the predicted summary:\n")
    #     for metric_name, value in bert_score_metrics.items():
    #         fout.write(f"{metric_name}: {value:.4f}\n")
    #         evaluation_metrics[metric_name] = value
    
    # Compute Summa-C scores
    avg_summac_score = mean_score(summac_scores)
    evaluation_metrics["summac_score"] = avg_summac_score
    # with open(log_path, "a") as fout:
    #     fout.write(f"Summa-C score: {avg_summac_score:.4f}\n")

    print(evaluation_metrics)
    with open(log_path, "a") as fout:
        fout.write(json.dumps(evaluation_metrics) + "\n")
    # TODO: Store all the metrics as data frame, save as csv?