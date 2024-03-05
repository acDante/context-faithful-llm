# Test CAD guided by important sentences on extractive summarisation data

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


def get_prompt(doc, important_sents, schema):
    # Prompt format: instruction + doc + important sentences
    prompt = ""
    if schema == "base":
        instruction = "Summarise the document below:"
        prompt = f"{instruction}\n\n{doc}\n\nSummary:"

    elif schema == "impt":
        instruction = "Summarise the document below:"
        prompt = f"{instruction}\n\n{doc}\n\nMain points:\n"
        for id, sent in enumerate(important_sents):
            prompt += f"{str(id + 1)}. {sent}\n"
        
        prompt += "\nSummary:"
    
    return prompt

def get_question_prompt(doc, schema):
    # Prompt format: instruction + doc
    prompt = ""
    if schema == "base":
        prompt = "Summary:"

    elif schema == "impt":
        instruction = "Summarise the document below:"
        prompt = f"{instruction}\n\n{doc}\n\nSummary:"
    
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
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test samples to evaluate on")
    parser.add_argument("--use_cad", action="store_true", help="Use context-aware decoding")
    parser.add_argument("--alpha", default=0.5, type=float, help="Parameter for context-aware decoding")
    parser.add_argument("--log_path", default="results/", type=str)
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--schema", default="base", type=str, choices=['base', 'impt'],
                        help="whether to include important sentences in the prompt")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    access_token = "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"
    download_path = "/home/hpcdu1/experiments/huggingface-hub"

    model_name = args.model_name
    base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map='auto',
                                                      torch_dtype=torch.bfloat16,
                                                      token=access_token,
                                                      cache_dir=download_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    model = base_model
    if args.use_cad:
        model = CAD(model=base_model, tokenizer=tokenizer)
    
    test_data = datasets.load_dataset("eReverter/cnn_dailymail_extractive", split="test")
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))  # Use the first few samples for faster iteration

    pred_samples = []
    predictions = []
    references = []

    for idx, sample in tqdm(enumerate(test_data)):
        article = sample['src']
        highlights = sample['tgt']  # reference summary
        labels = sample['labels']  # annotated extractive summary

        doc = " ".join(article)
        summary = " ".join(highlights)
        important_sents = [article[idx] for idx in range(len(labels)) if labels[idx]== 1]
        prompt = get_prompt(doc, important_sents, args.schema)
        print(prompt)
        
        # Build the input prompt
        if "chat" in args.model_name:
            messages = [
                {"role": "user", "content": prompt}
            ]
            input_ids = tokenizer.apply_chat_template(messages, truncation=True, return_tensors="pt")
            input_ids = input_ids.to("cuda")
        
        else:
            input_ids = tokenizer(prompt,
                                  max_length=4096,
                                  truncation=True,
                                  return_tensors="pt").input_ids.cuda("cuda")
        # Generate summary
        if args.use_cad:
            question_prompt = get_question_prompt(doc, args.schema)
            if "chat" in args.model_name:
                ques_messages = [
                    {"role": "user", "content": question_prompt}
                ]
                question_ids = tokenizer.apply_chat_template(ques_messages, truncation=True, return_tensors="pt")
                question_ids = question_ids.to("cuda")
            else:
                question_ids = tokenizer(question_prompt,
                                         max_length=4096,
                                         truncation=True,
                                         return_tensors="pt").input_ids.cuda("cuda")    
            
            raw_output = model.generate(texts=question_prompt,
                                        texts_with_context=prompt,
                                        alpha=args.alpha,
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
        references.append(summary)

        # Save predictions to file
        pred_sample = dict()
        pred_sample['article'] = doc
        pred_sample['highlights'] = summary
        pred_sample['important_sents'] = important_sents
        pred_sample['prediction'] = output
        pred_samples.append(pred_sample)
    
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
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metrics = {
        'rouge1': mean_score(rouge_scores['rouge1']),
        'rouge2': mean_score(rouge_scores['rouge2']),
        'rougeLsum': mean_score(rouge_scores['rougeLsum']),
        'bertscore_p': mean_score(bert_score_res['precision']),
        'bertscore_r': mean_score(bert_score_res['recall']),
        'bertscore_f1': mean_score(bert_score_res['f1']),
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