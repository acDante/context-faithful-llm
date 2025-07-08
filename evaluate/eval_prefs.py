import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import nltk
import numpy as np
import os
import copy

from summac.model_summac import SummaCZS, SummaCConv
from prefs.atomic_facts import AtomicFactGenerator
from prefs.factscorer import FactScorer


input_key = {
    "xsum": "document",
    "cnn_dm": "article",
    "ccsum": "article",
    "summscreen": "input",
    "qmsum": "input"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights",
    "ccsum": "summary",
    "summscreen": "output",
    "qmsum": "output"
}

def mean_score(scores):
    return sum(scores) / len(scores)

def compute_factscore(pred, gold, afg, fs):
    # Extract atomic facts from prediction and gold summary
    predicted_facts_and_sources = afg.extract_facts(pred)
    predicted_facts = [x for item in predicted_facts_and_sources for x in item[1]]
    predicted_facts = filter_facts(predicted_facts)

    fact_precision, fs_score_per_fact = fs.get_score(predicted_facts,
                                                     gold,
                                                     summname='test-summary',
                                                    )

    # Handle corner cases
    if len(predicted_facts) == 0 or np.isnan(fact_precision):
        fact_precision = 0.0

    metrics = {"fact_precision": fact_precision}

    return metrics, predicted_facts, fs_score_per_fact

def compute_prefscore(pred, gold, afg, fs):
    # Extract atomic facts from prediction and gold summary
    # afg = AtomicFactGenerator(model_name, cache_dir_prefix)
    predicted_facts_and_sources = afg.extract_facts(pred)
    predicted_facts = [x for item in predicted_facts_and_sources for x in item[1]]
    predicted_facts = filter_facts(predicted_facts)

    gold_facts_and_sources = afg.extract_facts(gold) # list of (sent, facts) tuples
    gold_facts = [x for item in gold_facts_and_sources for x in item[1]]
    gold_facts = filter_facts(gold_facts)   

    # Compute fact precision, fact recall and PREFS score
    # fs = FactScorer(model_name=model_name, cache_dir_prefix=cache_dir_prefix)
    fact_precision, fs_score_per_fact = fs.get_score(predicted_facts,
                                                     gold,
                                                     summname='test-summary',
                                                    )
    # Handle corner cases
    if len(predicted_facts) == 0 or np.isnan(fact_precision):
        fact_precision = 0.0

    fact_recall, score_per_fact = fs.get_score(gold_facts,
                                               pred,
                                               summname='test-summary',
                                               )
    # Handle corner cases
    if len(gold_facts) == 0 or np.isnan(fact_recall):
        fact_recall = 0.0
    
    if fact_precision + fact_recall == 0:
        prefs_score = 0.0
    
    else:
        prefs_score = (2 * fact_precision * fact_recall) / (fact_precision + fact_recall)

    metrics = {"fact_precision": fact_precision, 
               "fact_recall": fact_recall, 
               "prefs_score": prefs_score}
    
    return metrics, predicted_facts, gold_facts, fs_score_per_fact

def filter_facts(facts):
    """Some facts are vacuous of any real information, exclude these before scoring."""
    bad_substrings = ['someone','something','somebody','is a person','is a character', 'are people', 'are characters']
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or len(f.split())>2]
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or (not any(x in f.lower() for x in bad_substrings))]
    # facts = [f for f in facts if not f.lower().startswith('there is a') and not 'is in a room' in f and not 'is talking' in f and not 'are talking' in f and not 'made a statement' in f]
    facts = [f for f in facts if not 'is mentioned' in f.lower() and not 'are mentioned' in f.lower() and not 'is there' in f.lower() and not 'are there' in f.lower()]
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or not f.endswith(' to')]
    facts = [f for f in facts if not 'is there' in f and not 'are there' in f]
    return facts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the prediction file (.json)")
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum', 'summscreen', 'qmsum'])
    parser.add_argument("--metrics", type=str, choices=["summac", "factscore", "prisma"], default="factscore", help="Which evaluation metrics to compute")
    parser.add_argument("--log_path", type=str, help="Path to save the evaluation results for each file")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model used for computing FactScore")
    parser.add_argument("--exp_dir", type=str, default=".", help="Store cache files and evaluation metrics in this directory")

    args = parser.parse_args()
    return args

def extract_filename(json_path):
    # Get the basename (filename with extension)
    basename = os.path.basename(json_path)

    # Split the basename and extension
    filename, _ = os.path.splitext(basename)
    return filename

if __name__ == "__main__":

    args = parse_args()
    # Extract short model name
    short_model_name = {
        "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
        "meta-llama/Llama-3.3-70B-Instruct": "llama3.3-70b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
        "Qwen/Qwen2.5-72B-Instruct-AWQ": "qwen2.5-72b",
        "gpt-4o-mini": "gpt-4o-mini",
        "Qwen/Qwen3-32B": "qwen3-32b"
    }

    # Model for computing Summa-C scores
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda:0", start_file="default", agg="mean")
    
    data_path = args.data_path
    # result_path = Path("/home/xiaotang/Project/context-faithful-llm/guided-cad/results/")
    # data_path = result_path / "summary/mistral-7b/xsum-mistral-7b-base_preds.json"
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    
    documents = []
    golds = []
    predictions = []
    
    # Collect metrics: 
    summac_scores = []
    fact_scores = []
    prisma_scores = []
    
    cache_dir_prefix = os.path.join("exps", extract_filename(args.data_path))
    cache_dir_prefix = f"{cache_dir_prefix}_{short_model_name[args.model_name]}"
    if not os.path.exists(cache_dir_prefix):
        os.makedirs(cache_dir_prefix)

    # Initiate models for computing FactScore
    afg = AtomicFactGenerator(args.model_name, cache_dir_prefix)
    fs = FactScorer(model_name=args.model_name, cache_dir_prefix=cache_dir_prefix)

    annotated_samples = []
    for idx, sample in tqdm(enumerate(data)):
        document = sample[input_key[args.dataset]]
        gold_summary = sample[output_key[args.dataset]]
        # gold_summary = sample['summary']
        example_output = sample['generated_summary']
        annotated_sample = copy.deepcopy(sample)

        # Compute fact scores (TODO: debug nan, check the length/variables/fact_recall, e.g. using xsum-mistral-7b-base_preds.json)
        if args.metrics == "factscore":
            fact_score, predicted_facts, score_per_fact = compute_factscore(
                example_output,
                document,
                afg,
                fs
            )
            fact_scores.append(fact_score["fact_precision"])
            annotated_sample["fact_precision"] = fact_score["fact_precision"]
            annotated_sample["predicted_facts"] = predicted_facts
            annotated_sample["score_per_fact"] = score_per_fact

        if args.metrics == "prisma":
        #    fact_score = compute_factscore(example_output, gold_summary)
            fact_score, predicted_facts, gold_facts, score_per_fact = compute_prefscore(
                example_output, 
                document, 
                afg,
                fs
            )
            fact_scores.append(fact_score["fact_precision"])
            prisma_scores.append(fact_score["prefs_score"])
            annotated_sample["fact_precision"] = fact_score["fact_precision"]
            annotated_sample["prefs_score"] = fact_score["prefs_score"]
            annotated_sample["predicted_facts"] = predicted_facts
            # annotated_sample["gold_facts"] = gold_facts
            annotated_sample["score_per_fact"] = score_per_fact

        # Compute Summa-C score
        if args.metrics == "summac":
            summac_score = model_conv.score([document], [example_output])
            summac_scores.append(summac_score["scores"][0])
            annotated_sample["summac_score"] = summac_score["scores"][0]

        # Collect predictions and gold summaries
        documents.append(document)
        golds.append(gold_summary)
        predictions.append(example_output)

        # Store the evaluation metrics for each samples
        annotated_samples.append(annotated_sample)

    exp_dir = os.path.join("exps", extract_filename(args.data_path))
    exp_dir = f"{exp_dir}_{short_model_name[args.model_name]}"
    log_path = os.path.join(exp_dir, f"{args.metrics}.log")
    # with open(log_path, "a") as fout:
    #     fout.write(f"Currently evaluating: {args.data_path}\n")

    evaluation_metrics = {}
    file_path = os.path.basename(args.data_path)
    evaluation_metrics["exp_name"] = os.path.splitext(file_path)[0]

    # Compute Summa-C scores
    # summac_scores = model_conv.score(documents, predictions)
    if args.metrics == "summac" and len(summac_scores) > 0:
        avg_summac_score = mean_score(summac_scores)
        print(avg_summac_score)
        
        with open(log_path, "a") as fout:
            fout.write(f"Summa-C score: {avg_summac_score}\n")
    
    # Compute fact score
    if args.metrics == "factscore":
        avg_fact_score = mean_score(fact_scores)
        print("Fact Precision:", avg_fact_score)
        evaluation_metrics["fact_precision"] = avg_fact_score

        with open(log_path, "a") as fout:
            fout.write(json.dumps(evaluation_metrics) + "\n")


    # Compute fact scores and PRISMA score
    if args.metrics == "prisma":
        avg_fact_score = mean_score(fact_scores)
        avg_prisma_score = mean_score(prisma_scores)
        print("Fact Precision:", avg_fact_score)
        assert(len(prisma_scores) > 0)
        print("PRISMA Score:", avg_prisma_score)
        
        evaluation_metrics["fact_precision"] = avg_fact_score
        evaluation_metrics["prisma_score"] = avg_prisma_score

        with open(log_path, "a") as fout:
            fout.write(json.dumps(evaluation_metrics) + "\n")
            # fout.write(f"Fact Precision: {avg_fact_score}\n")
            # fout.write(f"PRISMA Score: {avg_prisma_score}\n")
    
    # Save the annotated samples
    annotated_samples_path = os.path.join(exp_dir, f"annotated_samples_{args.metrics}.json")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    with open(annotated_samples_path, "w") as fout:
        json.dump(annotated_samples, fout, indent=4)