# Extract top K attributed sentences by attention
import json
import argparse
from tqdm import tqdm
import copy

import torch
import numpy as np
from typing import List, Tuple, Dict
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login


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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum'])
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_samples", default=1000, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of most important sentences to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")
    parser.add_argument("--mode", default='last', type=str, choices=['last', 'mean'], help="How to aggregate the attention scores")

    args = parser.parse_args()
    return args

def load_data(dataset_name):
    if dataset_name == "extra_cnn":
        test_data = load_dataset("eReverter/cnn_dailymail_extractive", split="test")
    elif dataset_name == "cnn_dm":
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif dataset_name == 'xsum':
        test_data = load_dataset("xsum", split="test", trust_remote_code=True)
    elif dataset_name == "ccsum":
        # load CCSum test data (abstractive subset)
        ccsum_dataset = load_dataset("/mnt/ceph_rbd/datasets/ccsum")
        dataset_abstractive = ccsum_dataset.filter(lambda x: x["abstractiveness_bin"] == "high")
        test_data = dataset_abstractive['test']
    
    return test_data

def get_prompt_template(dataset_name):
    if dataset_name == "xsum":
        prompt_template = "Summarise the document below in one sentence:\n{context}"
    elif dataset_name == "cnn_dm":
        prompt_template = "Summarise the document below:\n{context}"
    elif dataset_name == "ccsum":
        prompt_template = "Summarise the document below in one sentence or two sentences:\n{context}"

    return prompt_template

def load_model_and_tokenzier(model_name="meta-llama/Llama-3.1-8B-Instruct", cache_dir="/mnt/ssd/llms"):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 output_attentions=True,
                                                 device_map="auto",
                                                 cache_dir=cache_dir,
                                                 )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_chat_prompt_and_ids(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_prompt_ids = tokenizer.encode(
        chat_prompt, add_special_tokens=False
    )

    return chat_prompt, chat_prompt_ids

def get_output(model, tokenizer, chat_prompt, chat_prompt_ids, max_response_tokens=128):
    t = torch.tensor([chat_prompt_ids])
    output_ids = model.generate(
        input_ids=t.to(model.device),
        max_new_tokens=max_response_tokens,
        do_sample=False,
        temperature=0.0,
    )[0]
    # We take the original prompt because sometimes encoding and decoding changes it
    raw_output = tokenizer.decode(output_ids)
    prompt_length = len(tokenizer.decode(chat_prompt_ids))
    output = chat_prompt + raw_output[prompt_length:]
    return output
    
def get_output_tokens(tokenizer, output_text):
    return tokenizer(output_text, add_special_tokens=False)


def prepare_input(document: str, model, tokenizer) -> Tuple[List[str], torch.Tensor]:
    """
    Prepare the input document by splitting it into sentences and tokenizing.
    Returns the sentences and tokenized inputs.
    """
    # Split document into sentences
    sentences, separators = split_into_sentences(document)
    
    # Create a conversation for summarization using the proper chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that generates concise summaries."},
        {"role": "user", "content": f"Please summarize the following text:\n\n{document}"}
    ]
    
    # Apply the chat template to format the conversation properly for the model
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    return sentences, inputs

def generate_summary(model, tokenizer, inputs, max_new_tokens: int = 128):
    original_input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask = inputs.attention_mask,
            max_new_tokens = max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            temperature=0.7,
            top_p=0.9
        )
    return outputs

# def compute_attention_scores(model, tokenizer, inputs, outputs):
#     attention_scores = outputs.attentions
#     attention_scores = [score[0][:, original_input_length:] for score in attention_scores]
    

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

def get_source_ranges(sentences, prompt="Summarise the document below: "):
    source_ranges = []
    cur_char_index = len(prompt)
    for sentence in sentences:
        source_ranges.append((cur_char_index, cur_char_index + len(sentence)))
        cur_char_index += len(sentence) + 1
    return source_ranges

def char_to_token(output_tokens, char_index):
    # output_tokens? char_index?
    for i in range(len(output_tokens["input_ids"]) - 1):
        if char_index < output_tokens.token_to_chars(i + 1).start:
            return i
    return i + 1

def get_source_token_ranges(tokenizer, output_tokens, source_ranges):
    # output_tokens? 
    # Find offset for chat template
    placeholder = "<placeholder>"
    messages = [{"role": "user", "content": placeholder}]
    placeholder_chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_offset_index = placeholder_chat_prompt.find(placeholder)

    source_token_ranges = []
    for start_index, end_index in source_ranges:
        token_start_index = char_to_token(
            output_tokens, start_index + chat_offset_index
        )
        token_end_index = (
            char_to_token(output_tokens, end_index + chat_offset_index - 1) + 1
        )
        source_token_ranges.append((token_start_index, token_end_index))
    return source_token_ranges

def get_response_start(chat_prompt_ids):
    return len(chat_prompt_ids)

def get_top_k_sentences(sentences, attention_scores, k=3):
    """Find the top K sentences with highest attention scores"""

    k = min(k, len(sentences))
    sentence_scores = list(zip(sentences, attention_scores))

    # Sort the sentences by attention scores in descending order
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    # Extract the top K sentences and their scores
    top_k_sentences = sorted_sentences[:k]
    
    # Separate sentences and scores
    top_sentences = [item[0] for item in top_k_sentences]
    top_scores = [item[1] for item in top_k_sentences]

    return top_sentences, top_scores

def main():

    args = parse_args()
    login("HF_TOKEN")

    # TODO: Add the main function (check: inseq_attention_llama3.1.ipynb) 
    # TODO: double check the hyperparameters (e.g. max_new_tokens, prompt format, make sure you use the right arguments when calling each utility function)
    model, tokenizer = load_model_and_tokenzier(model_name=args.model_name, cache_dir="/mnt/ceph_rbd/llms")
    test_data = load_data(args.dataset)
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    if args.dataset == "cnn_dm":
        max_new_tokens = 512
    else:
        max_new_tokens = 128
    
    skipped_samples = []
    processed_samples = []
    for idx, sample in tqdm(enumerate(test_data)):
        if idx % 100 == 0:
            print((f"Currently processing: {idx}-th sample.\n"))
        
        doc = sample[input_key[args.dataset]]
        prompt_template = get_prompt_template(args.dataset)
        prompt = prompt_template.format(context=doc)
        chat_prompt, chat_prompt_ids = get_chat_prompt_and_ids(prompt, tokenizer)

        try:
            output_text = get_output(model, tokenizer, chat_prompt, chat_prompt_ids, max_response_tokens=max_new_tokens)

            # 1. Extract attention scores (adapted from _get_attns() in ContextCite demo)
            # Attention scores = [num_layers, num_heads, num_output_tokens, num_output_tokens]
            output_tokens = get_output_tokens(tokenizer, output_text)
            response_start = get_response_start(chat_prompt_ids)  # Start of the generated token
        
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory during generation for sample {idx}. Skipping...")
            torch.cuda.empty_cache()
            skipped_samples.append(sample)
            continue

        device = model.device
        model.config.output_attentions = True
        batch = {
            "input_ids": torch.tensor(output_tokens["input_ids"]).to(device)[None],
            "attention_mask": torch.tensor(output_tokens["attention_mask"]).to(device)[
                None
            ],
            "labels": torch.tensor(
                [-100] * response_start + output_tokens["input_ids"][response_start:]
            ).to(device)[None],
        }

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True): # TODO: try inference_mode
            try:
                output = model(
                    **batch, 
                    output_attentions=True, 
                    output_hidden_states=False,
                    return_dict=True
                )

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory during attention computation for sample {idx}. Skipping...")
                torch.cuda.empty_cache()
                skipped_samples.append(sample)
                continue    

            # (1). Select the last layer to compute attention
            if args.mode == 'last':
                layers = (-1,)  # attention layers to use (use the last layer by default)
                attentions = torch.stack([output["attentions"][layer][0] for layer in layers]).cpu()  # [num_layers, num_heads, num_output_tokens, num_output_tokens]
            
            # (2). Alternatively, average the attention over all layers (note: this requires lots of GPU memory)
            elif args.mode == 'mean': 
                all_attentions = torch.stack([attn[0] for attn in output["attentions"]]).cpu()  # [num_layers, num_heads, num_output_tokens, num_output_tokens]
                attentions = all_attentions.mean(dim=0).unsqueeze(0)  # [num_heads, num_output_tokens, num_output_tokens]
            
            # print(attentions.shape)
            del output
            torch.cuda.empty_cache()   
        
        # Get attention scores for each (source, response_token) averaged over all heads/layers (adapted from _get_scores_by_response_token())
        sentences, _ = split_into_sentences(doc)
        source_ranges = get_source_ranges(sentences, prompt=prompt_template.format(context=""))  # Ensure you are using the same prompt sentence
        source_token_ranges = get_source_token_ranges(tokenizer, output_tokens, source_ranges)
        response_attns = attentions[:, :, response_start :]
        scores = [
            # TODO: Should this be mean or combination of mean and sum?
            response_attns[:, :, :, s:e].mean(dim=(0, 1, -1))
            for s, e in source_token_ranges
        ]
        scores_by_response_token = torch.stack(scores)  # [num_input_sentences x num_response_tokens]

        # Get attribution scores for ids range (aggregate attribution scores for output tokens), adapted from _get_attribution_scores_for_ids_range
        ids_start_index = 0
        ids_end_index = -1
        # scores = scores_by_response_token[:, ids_start_index:ids_end_index].mean(
        #     dim=-1
        # )
        scores = scores_by_response_token[:, ids_start_index:ids_end_index].max(
            dim=-1
        )[0]

        # Get the top K sentences with the highest attention scores
        top_k_sentences, top_k_scores = get_top_k_sentences(sentences, scores, k=args.num_sents)
        attributed_sents = []
        for sent, score in zip(top_k_sentences, top_k_scores):
            attributed_sents.append(
                {
                    "input_sequence": sent,
                    "score": score.item()
                }
            )
        
        # Save the attributed sentences and generated summary
        generated_summary = tokenizer.decode(output_tokens["input_ids"][response_start:])
        processed_sample = dict()
        processed_sample['id'] = sample['id']
        processed_sample[input_key[args.dataset]] = doc
        processed_sample[output_key[args.dataset]] = sample[output_key[args.dataset]]
        processed_sample.update({"attributed_sents": attributed_sents})
        processed_sample.update({"generated_summary": generated_summary})
        processed_samples.append(processed_sample)
    
    # Save the processed instances to JSON file
    with open(args.save_path, 'w') as fh:
        json.dump(processed_samples, fh, indent=4)
    
    if skipped_samples:
        skipped_save_path = args.save_path.replace('.json', '_skipped.json')
        with open(skipped_save_path, 'w') as fh:
            json.dump(skipped_samples, fh, indent=4)

if __name__ == "__main__":
    main()

# TODO: save the partial results when you get CUDA out of memory issue?
    
