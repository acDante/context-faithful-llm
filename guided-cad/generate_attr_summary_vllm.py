import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import copy
import nltk
from typing import List, Dict, Tuple
import re
import os
from dotenv import load_dotenv

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

def get_prompt_template(method, dataset):
    if method == "base":
        if dataset == "xsum":
            prompt_template = "Summarise the document below in one sentence:\n{doc}"
        elif dataset == "cnn_dm":
            prompt_template = "Summarise the document below:\n{doc}"
        elif dataset == "ccsum":
            # prompt_template = "Summarise the document below in one sentence or two sentences:\n{doc}"
            # prompt_template = "Generate an abstractive summary of the document below in one sentence:\n{doc}"
            prompt_template = "Summarize the following news article into one brief sentence: {doc}"

    elif method == "gen_attr":
        if dataset == "xsum":
            prompt_template = "Extract a list of {num_sents} key sentences from the input document and then generate a summary in one sentence only based on the extracted facts: {doc}\n\nHere is the output format.\nKey Sentences:\n1. sentence1, 2. sentence2, ...\nSummary:\n[summary]\n"
        elif dataset == "cnn_dm":
            prompt_template = "Extract a list of {num_sents} key sentences from the input document and then generate a summary only based on the extracted facts: {doc}\n\nHere is the output format.\nKey Sentences:\n1. sentence1, 2. sentence2, ...\nSummary:\n[summary]\n"
        elif dataset == "ccsum":
            # prompt_template = "Extract a list of {num_sents} key sentences from the input document and then generate a summary in one sentence or two sentences only based on the extracted facts: {doc}\n\nHere is the output format.\nKey Sentences:\n1. sentence1, 2. sentence2, ...\nSummary:\n[summary]\n"
            # prompt_template = "Extract a list of {num_sents} key sentences from the input document and then generate an abstractive summary in one sentence only based on the extracted facts: {doc}\n\nHere is the output format.\nKey Sentences:\n1. sentence1, 2. sentence2, ...\nSummary:\n[summary]\n"
            prompt_template = "Extract a list of {num_sents} key sentences from the input document and then generate a summary in one sentence only based on the extracted facts: {doc}\n\nHere is the output format.\nKey Sentences:\n1. sentence1, 2. sentence2, ...\nSummary:\n[summary]\n"

    return prompt_template

# def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", cache_dir="/mnt/ssd/llms"):
#     llm = LLM(model=model_name, download_dir=cache_dir)
#     return llm

def load_data(args):
    if args.dataset == "xsum":
        test_data = load_dataset("xsum", split="test", trust_remote_code=True)
    elif args.dataset == "cnn_dm":
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif args.dataset == "ccsum":
        # load CCSum test data (abstractive subset)
        ccsum_dataset = load_dataset("/mnt/ceph_rbd/datasets/ccsum")  # Use this on EIDF
        # ccsum_dataset = load_dataset("/home/xiaotang/Project/context-faithful-llm/datasets/ccsum")
        dataset_abstractive = ccsum_dataset.filter(lambda x: x["abstractiveness_bin"] == "high")
        test_data = dataset_abstractive['test']
        
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    return test_data

def split_into_sentences(text: str) -> Tuple[List[str], List[str]]:
    lines = text.splitlines()
    sentences = []
    for line in lines:
        sentences.extend(nltk.sent_tokenize(line))
    separators = []
    cur_start = 0
    for sentence in sentences:
        cur_end = text.find(sentence, cur_start)
        separators.append(text[cur_start:cur_end])
        cur_start = cur_end + len(sentence)
    return sentences, separators

def extract_sentences_and_summary(text):
    """
    Extract key sentences and summary from LLM output.
    
    Args:
        text (str): The LLM output text containing key sentences and summary
        
    Returns:
        tuple: (list of key sentences, summary string)
    """
    try:
        # Find the summary section
        summary_start = text.find("Summary:")
        
        if summary_start == -1:
            return [], ""
        
        # Extract everything before "Summary:" as key sentences text
        key_sentences_text = text[:summary_start].strip("\n[]")
        
        # Extract the summary
        summary = text[summary_start + len("Summary:"):].strip()
        
        # Parse the key sentences into a list
        sentences = []
        
        # Split the text into lines
        lines = key_sentences_text.split('\n')
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number followed by period (e.g., "1.")
            if re.match(r'^\d+\.', line):
                # Remove the number prefix
                sentence = re.sub(r'^\d+\.\s*', '', line).strip()
                if sentence:
                    sentences.append(sentence)
            # Also check for bullet points for backward compatibility
            elif line.startswith('- ') or line.startswith('* '):
                sentence = line[2:].strip()
                if sentence:
                    sentences.append(sentence)
        
        return sentences, summary
    
    except Exception as e:
        print(f"Error parsing text: {e}")
        return [], ""
    
# Post process the generated summary
def post_process(output_text, dataset):
    if dataset == "xsum":
        sentences = nltk.sent_tokenize(output_text)
        output_text = sentences[0] if sentences else ""
    
    elif dataset == "cnn_dm":
        sentence_end_pattern = r'[。！？\.!?]'
        sentences = nltk.sent_tokenize(output_text)
        complete_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if re.search(sentence_end_pattern + r'$', sentence):
                complete_sentences.append(sentence)

        output_text = " ".join(complete_sentences)
    
    elif dataset == "ccsum":
        if "\n\n" in output_text:  # Remove the prompt words generated by Llama3
            output_text = output_text.split("\n\n")[-1]

    return output_text
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dataset", default="xsum", type=str, help="name of the dataset to evaluate on")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test samples to evaluate on")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of attributed sentences to extract")
    parser.add_argument("--log_path", default="results/summary", type=str)
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="Maximum number of tokens to generate")
    parser.add_argument("--method", type=str, default="base", help="Which prompt template to use", choices=["base", "gen_attr"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_num_batched_tokens", type=int, default=1024)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    
    # Load test dataset
    test_data = load_data(args)

    # Load model with vLLM
    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        gpu_memory_utilization = args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,  # Reduce if OOM, increase for better throughput
        swap_space=4,
        enforce_eager=False,  # Keep as False for better performance
        download_dir="/mnt/ceph_rbd/llms"
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    log_path = Path(args.log_path)
    output_path = log_path / f"{args.exp_name}_preds.json"

    prompt_template = get_prompt_template(args.method, args.dataset)
    
    # Prepare all prompts for batch generation
    chat_messages = []
    for idx, sample in enumerate(test_data):
        doc = sample[input_key[args.dataset]]
        if args.method == "base":
            prompt = prompt_template.format(doc=doc)
        elif args.method == "gen_attr":
            prompt = prompt_template.format(num_sents=args.num_sents, doc=doc)
        
        chat_messages.append([
            {"role": "user", "content": prompt}
        ])
    
    # Generate all outputs in batch
    if "Qwen3" in args.model_name:
        outputs = llm.chat(messages=chat_messages,
                           sampling_params=sampling_params,
                           use_tqdm=True,
                           chat_template_kwargs={"enable_thinking": False})
    
    else:
        outputs = llm.chat(messages=chat_messages,
                           sampling_params=sampling_params,
                           use_tqdm=True)
    
    # Process results
    processed_samples = []
    for idx, (sample, output) in tqdm(enumerate(zip(test_data, outputs))):
        doc = sample[input_key[args.dataset]]
        output_text = output.outputs[0].text
        
        if args.method == "gen_attr":
            key_sentences, summary = extract_sentences_and_summary(output_text)
            summary = post_process(summary, args.dataset)

            # Save the attributed sentences and generated summary
            attributed_sents = []
            for sent in key_sentences:
                sent = sent.strip()
                if len(sent) > 0:
                    attributed_sents.append(
                        {
                            "input_sequence": sent,
                            "score": 1.0,
                        }
                    )

            processed_sample = dict()
            processed_sample['id'] = sample['id']
            processed_sample[input_key[args.dataset]] = doc
            processed_sample[output_key[args.dataset]] = sample[output_key[args.dataset]]
            processed_sample.update({"attributed_sents": attributed_sents})
            processed_sample.update({"generated_summary": summary})
            processed_sample.update({"raw_output": output_text})
            processed_samples.append(processed_sample)
        
        elif args.method == "base":
            summary = post_process(output_text, args.dataset)
            processed_sample = dict()
            processed_sample['id'] = sample['id']
            processed_sample[input_key[args.dataset]] = doc
            processed_sample[output_key[args.dataset]] = sample[output_key[args.dataset]]
            processed_sample.update({"generated_summary": summary})
            processed_sample.update({"raw_output": output_text})
            processed_samples.append(processed_sample)          
    
    # Save the processed instances to JSON file
    with open(output_path, 'w') as fh:
        json.dump(processed_samples, fh, indent=4)


if __name__ == "__main__":
    main()