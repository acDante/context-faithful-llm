# [improved version of evaluation script] Compute the ROUGE, BERT score and Summa-C score for the prediction
# Save the evaluation metric for each test sample

import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import nltk
import os
import copy

from summac.model_summac import SummaCZS, SummaCConv


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

def eval_rouge_scores_individual(pred, label):
    """Compute ROUGE scores for a single sample"""
    rouge = evaluate.load('rouge')
    processed_pred, processed_label = postprocess_text([pred], [label])
    rouge_scores = rouge.compute(predictions=processed_pred,
                                 references=processed_label)
    metrics = {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'rougeLsum': rouge_scores['rougeLsum']
    }
    return metrics

def eval_bert_scores_individual(pred, label):
    """Compute BERTScore for a single sample"""
    bert_score = evaluate.load('bertscore')
    bert_score_res = bert_score.compute(predictions=[pred], 
                                        references=[label], 
                                        model_type="microsoft/deberta-xlarge-mnli", 
                                        lang="en")
    metrics = {
        'bertscore_p': bert_score_res['precision'][0],
        'bertscore_r': bert_score_res['recall'][0],
        'bertscore_f1': bert_score_res['f1'][0],
    }
    return metrics

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
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum', 'summscreen', 'qmsum', 'gov_report'])
    parser.add_argument("--log_path", type=str, help="Path to save the evaluation results for each file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])

    # Model for computing Summa-C scores
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=f"cuda:0", start_file="default", agg="mean")

    data_path  = args.data_path
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    
    rougeL_scores = []
    bert_f1_scores = []
    summac_scores = []

    annotated_samples = []
    for idx, sample in tqdm(enumerate(data)):
        document = sample[input_key[args.dataset]]
        gold_summary = sample[output_key[args.dataset]]
        pred_summary = sample['generated_summary']
        annotated_sample = copy.deepcopy(sample)

        # Compute Summa-C score for each sample
        summac_score = model_conv.score([document], [pred_summary])
        summac_scores.append(summac_score["scores"][0])
        annotated_sample["summac_score"] = summac_score["scores"][0]

        # Compute ROUGE scores for each sample
        rouge_scores = eval_rouge_scores_individual(pred_summary, gold_summary)
        rougeL_scores.append(rouge_scores['rougeLsum'])
        annotated_sample["rougeL"] = float(rouge_scores["rougeLsum"])

        # Compute BERT scores for each sample
        bert_scores = eval_bert_scores_individual(pred_summary, gold_summary)
        bert_f1_scores.append(bert_scores['bertscore_f1'])
        annotated_sample["bertscore_f1"] = float(bert_scores["bertscore_f1"])

        annotated_samples.append(annotated_sample)
    
    log_path = os.path.join("logs", extract_filename(args.data_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    evaluation_metrics = {}
    file_path = os.path.basename(args.data_path)
    exp_name = os.path.splitext(file_path)[0]
    evaluation_metrics["exp_name"] = exp_name

    evaluation_metrics["rougeL"] = mean_score(rougeL_scores)
    evaluation_metrics["bertscore_f1"] = mean_score(bert_f1_scores)
    evaluation_metrics["summac_score"] = mean_score(summac_scores)

    print(evaluation_metrics)
    save_path = os.path.join(log_path, "average_metrics.json")
    if not os.path.exists(log_path):
        with open(save_path, "w") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
            fout.write('\n')
    else:
        with open(save_path, "a") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
            fout.write('\n')
    # with open(os.path.join(log_path, "average_metrics.json"), "w") as fout:
    #     json.dump(evaluation_metrics, fout, indent=4)
    
    with open(os.path.join(log_path, "metrics_per_sample.json"), "w") as fout:
        json.dump(annotated_samples, fout, indent=4)
