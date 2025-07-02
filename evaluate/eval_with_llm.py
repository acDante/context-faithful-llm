import json
import argparse
import sys
import os
import copy
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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

REL_PROMPT_TEMPLATE = '''You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-10) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 10.


Example:


Source Text:

{source}

Summary:

{target}


Evaluation Form (scores ONLY):

- Relevance: '''

CON_PROMPT_TEMPLATE = '''You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-10) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Example:


Source Text: 

{source}

Summary: 

{target}


Evaluation Form (scores ONLY):

- Consistency: '''


PROMPT_TEMPLATE = '''You will be given a source text. You will then be given one target text to be evaluated.

Your task is to rate the information alignment of target text against the source text.

Please make sure you read and understand these instructions carefully. Please keep this source text open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-10) - the information alignment between the target text and the source text. A consistent target text contains only statements that are entailed by the source text. Annotators were also asked to penalize target texts that contained hallucinated facts. 1 - worst, 5 - best.


Evaluation Steps:

1. Read the source text carefully and identify the main facts and details it presents.
2. Read the target text and compare it to the source text. Check if the target text contains any factual errors that are not supported by the source text.
4. Assign a score for consistency based on the Evaluation Criteria.

Note: only output the score for consistency, no other text.

Source Text: 

{source}

Target Text: 

{target}

Evaluation Form (scores ONLY):

- Consistency: '''

class QwenEvaluator:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto", 
            device_map="auto",
            cache_dir="/mnt/ceph_rbd/llms"
        )
       
    def evaluate(self, args, source, target):
        if args.metrics == "consistency":
            prompt = PROMPT_TEMPLATE.format(source=source, target=target)
        elif args.metrics == "relevance":
            prompt = REL_PROMPT_TEMPLATE.format(source=source, target=target)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False 
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
       
        generated_ids = self.model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=False,
        temperature=None, 
        top_p=None, 
        top_k=None, 
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        return content

def extract_consistency_score(output_text):
    if "Consistency" in output_text:
        score_text = output_text.lower().split("consistency:")[1].strip().split()[0]
    else:
        score_text = output_text
    
    numbers = re.findall(r'\d+', score_text)
    if numbers:
        return numbers[0]

def mean_score(scores):
    return sum(scores) / len(scores)

def extract_filename(json_path):
    # Get the basename (filename with extension)
    basename = os.path.basename(json_path)

    # Split the basename and extension
    filename, _ = os.path.splitext(basename)
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate target text against source text using Qwen model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Qwen model.")
    parser.add_argument("--data_path", type=str, help="Path to the prediction file (.json)")
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'ccsum'])
    parser.add_argument("--output_path", type=str, default="results/scores.json", help="Path to save the evaluation scores.")
    parser.add_argument("--metrics", type=str, choices=['consistency', 'relevance'], default='consistency', help="Which dimension to evaluate")
    
    args = parser.parse_args()
    evaluator = QwenEvaluator(args)

    data_path = args.data_path
    with open(data_path, 'r') as fin:
        data = json.load(fin)

    consistency_scores = []
    annotated_samples = []

    for idx, sample in tqdm(enumerate(data)):
        document = sample[input_key[args.dataset]]
        gold_summary = sample[output_key[args.dataset]]
        pred_summary = sample['generated_summary']
        annotated_sample = copy.deepcopy(sample)

        # Evaluate consistency using Qwen model
        score = evaluator.evaluate(args, document, pred_summary)
        score = extract_consistency_score(score)
        # print("Evaluating sample {}: {}".format(idx, score))
        consistency_scores.append(int(score))
        annotated_sample[args.metrics] = int(score)
        
        annotated_samples.append(annotated_sample)
    
    log_path = os.path.join("logs", extract_filename(args.data_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    evaluation_metrics = {}
    file_path = os.path.basename(args.data_path)
    exp_name = os.path.splitext(file_path)[0]
    evaluation_metrics["exp_name"] = exp_name
    evaluation_metrics[args.metrics] = mean_score(consistency_scores)

    print(evaluation_metrics)
    # Check if average_metrics.json already exsits, if not exsit, us "w" mode otherwise use "a" mode
    save_path = os.path.join(log_path, "average_metrics.json")
    if not os.path.exists(log_path):
        with open(save_path, "w") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
    else:
        with open(save_path, "a") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
    
    with open(os.path.join(log_path, f"qwen_{args.metrics}_per_sample.json"), "w") as fout:
        json.dump(annotated_samples, fout, indent=4)

