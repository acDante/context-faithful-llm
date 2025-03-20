# Extract top K attributed sentences by ContextCite method
import json
import argparse
from tqdm import tqdm
import copy

import torch
from context_cite import ContextCiter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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

def load_data(dataset_name):
    if dataset_name == "extra_cnn":
        test_data = load_dataset("eReverter/cnn_dailymail_extractive", split="test")
    elif dataset_name == "cnn_dm":
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif dataset_name == 'xsum':
        test_data = load_dataset("xsum", split="test")
    elif dataset_name == "ccsum":
        # load CCSum test data (abstractive subset)
        ccsum_dataset = load_dataset("/mnt/ceph_rbd/datasets/ccsum")
        dataset_abstractive = ccsum_dataset.filter(lambda x: x["abstractiveness_bin"] == "high")
        test_data = dataset_abstractive['test']
    
    return test_data

def load_model(model_name, cache_dir="/mnt/ceph_rbd/llms", device="cuda"):
    config = AutoConfig.from_pretrained(model_name)
    context_window_length = getattr(config, 'max_position_embeddings', 
                                    getattr(config, 'n_positions', None))
    
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 cache_dir=cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              cache_dir=cache_dir)
    tokenizer.model_max_length = context_window_length
    
    return model, tokenizer

def get_prompt_template(dataset_name):
    if dataset_name == "xsum":
        prompt_template = "Summarise the document below in one sentence:\n{context}"
    elif dataset_name == "cnn_dm":
        prompt_template = "Summarise the document below:\n{context}"
    elif dataset_name == "ccsum":
        prompt_template = "Summarise the document below in one sentence or two sentences:\n{context}"

    return prompt_template

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum'])
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_samples", default=1000, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of most important sentences to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    login("HF_TOKEN")

    # Load the test data
    model, tokenizer = load_model(
        args.model_name, cache_dir="/mnt/ceph_rbd/llms", device="cuda"
    )
    test_data = load_data(args.dataset)
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    if args.dataset == "cnn_dm":
        max_new_tokens = 256
    else:
        max_new_tokens = 128

    processed_samples = []
    for idx, sample in tqdm(enumerate(test_data)):
        if idx % 100 == 0:
            print(f"Currently processing: {idx}-th sample.\n")
        
        context = sample[input_key[args.dataset]]
        query = "Generate a summary of the document"

        # Extract top K attributed sentences by ContextCiter
        cc = ContextCiter(model, tokenizer, context, query)
        cc.prompt_template = get_prompt_template(args.dataset)
        cc.generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0
        }

        results = cc.get_attributions(as_dataframe=True, top_k=args.num_sents)
        df = results.data

        attributed_sents = []
        for index, row in results.data.iterrows():
            attributed_sents.append(
                {
                    "input_sequence": row['Source'],
                    "score": row['Score']
                }
            )

        # Save the attributed sentences and generated summary
        processed_sample = dict()
        processed_sample['id'] = sample['id']
        processed_sample[input_key[args.dataset]] = context
        processed_sample[output_key[args.dataset]] = sample[output_key[args.dataset]]
        processed_sample.update({"attributed_sents": attributed_sents})
        processed_sample.update({"generated_summary": cc.response})
        
        processed_samples.append(processed_sample)

    # Save the processed instances to a JSON file
    with open(args.save_path, 'w') as fh:
        json.dump(processed_samples, fh, indent=4)

if __name__ == "__main__":
    main()
    