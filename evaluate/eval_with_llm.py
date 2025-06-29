import json
import argparse
import sys
import os
import copy
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

PROMPT_TEMPLATE = '''You will be given a source text. You will then be given one target text to be evaluated.

Your task is to rate the information alignment of target text against the source text.

Please make sure you read and understand these instructions carefully. Please keep this source text open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the information alignment between the target text and the source text. A consistent target text contains only statements that are entailed by the source text. Annotators were also asked to penalize target texts that contained hallucinated facts. 1 - worst, 5 - best.


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

- Consistency:'''

class QwenEvaluator:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto", 
            device_map="auto",
            cache_dir="/mnt/ssd/llms"
        )
       
    def evaluate(self, source, target):
        prompt = PROMPT_TEMPLATE.format(source=source, target=target)
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

def extract_filename(json_path):
    # Get the basename (filename with extension)
    basename = os.path.basename(json_path)

    # Split the basename and extension
    filename, _ = os.path.splitext(basename)
    return filename

def mean_score(scores):
    return sum(scores) / len(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate target text against source text using Qwen model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Qwen model.")
    parser.add_argument("--data_path", type=str, help="Path to the prediction file (.json)")
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'ccsum'])
    parser.add_argument("--output_path", type=str, default="results/scores.json", help="Path to save the evaluation scores.")
    
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
        score = evaluator.evaluate(document, pred_summary)
        print("Evaluating sample {}: {}".format(idx, score))
        consistency_scores.append(int(score))
        annotated_sample["qwen_score"] = int(score)
        
        annotated_samples.append(annotated_sample)
    
    log_path = os.path.join("logs", extract_filename(args.data_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    evaluation_metrics = {}
    file_path = os.path.basename(args.data_path)
    exp_name = os.path.splitext(file_path)[0]
    evaluation_metrics["exp_name"] = exp_name
    evaluation_metrics["qwen_score"] = mean_score(consistency_scores)

    print(evaluation_metrics)
    # Check if average_metrics.json already exsits, if not exsit, us "w" mode otherwise use "a" mode
    save_path = os.path.join(log_path, "average_metrics.json")
    if not os.path.exists(log_path):
        with open(save_path, "w") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
    else:
        with open(save_path, "a") as fout:
            json.dump(evaluation_metrics, fout, indent=4)
    
    with open(os.path.join(log_path, "llm_metrics_per_sample.json"), "w") as fout:
        json.dump(annotated_samples, fout, indent=4)

    print(f"Evaluation Score: {score}")
