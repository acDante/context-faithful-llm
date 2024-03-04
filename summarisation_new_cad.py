"""Test alternative CAD implementation"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import datasets
import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from models import context_aware_wrapper, CADModel
import nltk
from metrics import mean_score
from cad import CAD


# TODO: set maximum length for each model?
input_key = {
    "xsum": "document",
    "cnn_dm": "article"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights"
}

# TODO: make the model to focus on both the context and instructions?
# TODO: tune the alpha values
def get_question_prompt(doc, schema, summary=""):
    if schema == "base":
        prompt = f'Summary:{summary}'
    elif schema == "opin":
        prompt = f'Summarise the above article in one sentence in the narrator\'s opinion. Summary:{summary}'
    elif schema == "attr":
        prompt = f'Summarize the article in one sentence. Summary:{summary}'
    elif schema == 'instr':
        prompt = f'Summary:{summary}'
    elif schema == 'instr+opin':
        prompt = f'Summarise the above article in the narrator\'s opinion. Summary:{summary}'

    return prompt

def get_prompt(doc, schema, summary=""):
    """Context faithful prompting for summarisation"""
    prompt = ""
    if schema == "base":
        prompt = f'Document: {doc}\nSummary:{summary}'
    elif schema == "opin":
        prompt = f'A narrator said "{doc}"\nSummarise the above article in one sentence in the narrator\'s opinion. Summary:{summary}'
    elif schema == "attr":
        prompt = f'Document: {doc}\nSummarize the article in one sentence. Summary:{summary}'
    elif schema == 'instr':
        prompt = f'Instruction: read the given article and provide a summary in one sentence.\n\nDocument: {doc}\nSummary:{summary}'
    elif schema == 'instr+opin':
        prompt = f'Instruction: read the given article and provide a summary in one sentence.\n\A narrator said "{doc}"\nSummarise the above article in the narrator\'s opinion. Summary:{summary}'
    elif schema == 'instr-brief':    # Rohit's prompt  # TODO: need to change tokenisation codes
        prompt = f"## {doc} \n ## 100 WORD SUMMARY \n"

    return prompt

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
    parser.add_argument("--dataset", default="cnn_dm", type=str, choices=['cnn_dm', 'xsum'])
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test samples to evaluate on")
    parser.add_argument("--use_cad", action="store_true", help="Use context-aware decoding")
    parser.add_argument("--alpha", default=1.0, type=float, help="Parameter for context-aware decoding")
    parser.add_argument("--schema", default="base", type=str, choices=['base', 'opin', 'instr', 'attr', 'instr+opin', 'instr-brief'],
                        help="which context faithful prompting format to use")
    parser.add_argument("--log_path", default="results/", type=str)
    parser.add_argument("--exp_name", type=str, help="Experiment name")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    access_token = "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"
    download_path = "/home/hpcdu1/experiments/huggingface-hub"

    # TODO: try vicuna model and Rohit's prompt
    # lmsys/vicuna-13b-v1.5 

    model_name = args.model_name
    base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map="auto",  # Q: Note: this code cannot work on multiple GPUs! How to support device_map="auto" for custom model?
                                                      torch_dtype=torch.bfloat16,
                                                      token=access_token,
                                                      cache_dir=download_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False,
                                              padding_side="left",
                                              token=access_token,
                                              cache_dir=download_path)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    # >>> TODO: compare with CAD implementation on github <<<
    model = base_model
    if args.use_cad:
        model = CAD(model=base_model, tokenizer=tokenizer)

    # Load FactKB model
    factkb_tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True,
                                                     cache_dir=download_path)
    factkb_model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2, 
                                                                      device_map="auto", cache_dir=download_path)

    if args.dataset == "cnn_dm":
        test_data = datasets.load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif args.dataset == 'xsum':
        test_data = datasets.load_dataset("xsum", split="test")

    test_data = test_data.select(range(min(args.num_samples, len(test_data))))  # Use the first few samples for faster iteration

    factkb_scores = []
    pred_samples = []
    predictions = []
    references = []

    # TODO: this prompt format only works for chat model
    # Q: test data contains about 10k samples, how to speed up inference?
    for idx, sample in tqdm(enumerate(test_data)):
        doc = sample[input_key[args.dataset]]
        reference = sample[output_key[args.dataset]]
        # Q: what prompt format to use?
        prompt = get_prompt(doc, args.schema)
        messages = [
            {"role": "user", "content": prompt}
        ]
        if "chat" in args.model_name:
            input_ids = tokenizer.apply_chat_template(messages, truncation=True, return_tensors="pt")
            input_ids = input_ids.to("cuda")

        else:
            # Prompt for Llama / Llama-2 non-chat models
            input_ids = tokenizer(prompt,
                                  max_length=4096,
                                  truncation=True,
                                  return_tensors="pt").input_ids.cuda("cuda")
            # input_ids = input_ids.to(model.device)     

        # TODO: >>>> separate the inference and evaluation steps <<<<

        # TODO: Check CAD implementation, be careful with special tokens in the prompt!
        # Issue: model tends to generate a very long summary. when to stop?
        # Q: How to set proper hyperparameters for summarisation?
        # TODO: try stopping criteria
        if args.use_cad:
            ques_messages = [
                {"role": "user", "content": get_question_prompt(doc, args.schema)}
            ]
            if "chat" in args.model_name:
                question_ids = tokenizer.apply_chat_template(ques_messages, truncation=True, return_tensors="pt")
                question_ids = question_ids.to("cuda")
            else:
                question_prompt = get_question_prompt(doc, args.schema)
                question_ids = tokenizer(question_prompt,
                                         max_length=4096,
                                         truncation=True,
                                         return_tensors="pt").input_ids.cuda("cuda")

            question_prompt = get_question_prompt(doc, args.schema)
            raw_output = model.generate(texts=question_prompt,
                                        texts_with_context=prompt,
                                        do_sample=False,
                                        max_new_tokens=128,
                                        temperature=0.0)
            output = raw_output[0]

        else:
            outputs = model.generate(input_ids,
                                     do_sample=False,
                                     max_new_tokens=128,
                                     temperature=0.0)
            
            raw_output = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
            output = raw_output

        predictions.append(output)
        references.append(reference)
        
        # Save predictions to file
        pred_sample = dict()
        pred_sample['article'] = doc
        pred_sample['highlights'] = reference
        pred_sample['prediction'] = output
        pred_samples.append(pred_sample)

        # Evaluate FactKB score
        factkb_input = [[output, doc]]
        factkb_tokens = factkb_tokenizer(factkb_input, return_tensors="pt", 
                                         padding="max_length", truncation=True).to(factkb_model.device)
        factkb_logits = factkb_model(**factkb_tokens).logits
        factkb_res = torch.softmax(factkb_logits, dim=1)
        factkb_scores.append(float(factkb_res[0][1]))

    # Evaluate the performance:  https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/blob/main/src/backend/tasks/xsum/task.py#L55

    # TODO: compute the evaluation metrics for each instance and then collate together?
    # Compute the ROUGE scores -> Q: which ROUGE implementation to use?
    rouge = evaluate.load('rouge')
    processed_preds, processed_refs = postprocess_text(predictions, references)
    rouge_scores = rouge.compute(predictions=processed_preds, 
                                 references=processed_refs,
                                 use_aggregator=False)

    # Compute BERT score
    bert_score = evaluate.load('bertscore')
    bert_score_res = bert_score.compute(predictions=predictions, 
                                        references=references, 
                                        model_type="microsoft/deberta-xlarge-mnli", lang="en")

    # Aggregate the evaluation metrics
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metrics = {
        'rouge1': mean_score(rouge_scores['rouge1']),
        'rouge2': mean_score(rouge_scores['rouge2']),
        'rougeLsum': mean_score(rouge_scores['rougeLsum']),
        'bertscore_p': mean_score(bert_score_res['precision']),
        'bertscore_r': mean_score(bert_score_res['recall']),
        'bertscore_f1': mean_score(bert_score_res['f1']),
        "factKB": mean_score(factkb_scores)
    }
    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}_preds.json", 'w') as fh:
            json.dump(pred_samples, fh, indent=4)

    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}.log", "w") as fout:
            for metric_name, value in metrics.items():
                fout.write(f"{metric_name}: {value:.4f}")
                fout.write("\n")


if __name__ == '__main__':
    main()
