# Extract top K attributed sentences by ContextCite method
import json
import argparse
from tqdm import tqdm
import copy

import torch
from context_cite.context_citer import ContextCiter
from context_cite.context_citer import Qwen3ContextCiter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv


input_key = {
    "xsum": "document",
    "cnn_dm": "article",
    "ccsum": "article",
    "summscreen": "input",
    "gov_report": "input",
    "qmsum": "input"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights",
    "ccsum": "summary",
    "summscreen": "output",
    "gov_report": "output",
    "qmsum": "output"
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
    elif dataset_name == "gov_report":
        test_data = load_dataset("tau/scrolls", dataset_name)["validation"]
    elif dataset_name == "summscreen":
        test_data = load_dataset("tau/scrolls", "summ_screen_fd")["validation"]
    elif dataset_name == "qmsum":
        test_data = load_dataset("tau/scrolls", "qmsum")["validation"]
    
    return test_data

def load_model(model_name, cache_dir="/mnt/ceph_rbd/llms", device="cuda"):
    config = AutoConfig.from_pretrained(model_name)
    context_window_length = getattr(config, 'max_position_embeddings', 
                                    getattr(config, 'n_positions', None))
    
    if "Qwen" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype="auto",
                                                     device_map="auto",
                                                     cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    cache_dir=cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              cache_dir=cache_dir)
    # tokenizer.padding_side = "left"
    tokenizer.model_max_length = context_window_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_prompt_template(dataset_name):
    if dataset_name == "xsum":
        prompt_template = "Summarise the document below in one sentence:\n{context}"
    elif dataset_name == "cnn_dm":
        prompt_template = "Summarise the document below:\n{context}"
    elif dataset_name == "ccsum":
        # prompt_template = "Summarise the document below in one sentence or two sentences:\n{context}"
        # prompt_template = "Generate an abstractive summary of the document below in one sentence:\n{context}"
        prompt_template = "Summarize the following news article into one brief sentence:\n{context}"
    elif dataset_name == "gov_report":
        prompt_template = "You are given a report by a government agency. Write a one-page summary of the report. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n\nReport:\n{context}"
    elif dataset_name == "summscreen":
        prompt_template = "Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{context}\nNow generate the summary in 5 sentences:"
    elif dataset_name == "qmsum":
        prompt_template = "Read the following meeting transcript. Produce a summary in 4 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{context}\nNow generate the summary in 4 sentences:"

    return prompt_template

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum', 'summscreen', 'gov_report', 'qmsum'])
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_samples", default=1000, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of most important sentences to extract")
    parser.add_argument("--num_ablations", default=64, type=int, help="The number of ablations used to train the surrogate model.")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")
    parser.add_argument("--shard", type=int, default=None, help="Process different shards in parallel")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    load_dotenv("../.env")
    hf_token = os.environ.get("HF_TOKEN")
    login(hf_token)

    # Load the test data
    model, tokenizer = load_model(
        args.model_name, cache_dir="/mnt/ceph_rbd/llms", device="cuda"
    )

    # Debug: try to optimise inference [ADD]
    # model.generation_config.cache_implementation = "static"
    # tokenizer.padding_side = "left"
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    test_data = load_data(args.dataset)
    if args.shard is not None:
        start_idx = args.shard * args.num_samples
        end_idx = min(start_idx + args.num_samples. len(test_data))
        test_data = test_data.select(range(start_idx, end_idx))
    else:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    if args.dataset == "cnn_dm":
        max_new_tokens = 512
    elif args.dataset == "gov_report":
        max_new_tokens = 1024
    elif args.dataset == "summscreen":
        max_new_tokens = 512
    elif args.dataset == "qmsum":
        max_new_tokens = 512
    else:
        max_new_tokens = 128

    skipped_samples = []
    processed_samples = []
    for idx, sample in tqdm(enumerate(test_data)):
        if idx % 100 == 0:
            print(f"Currently processing: {idx}-th sample.\n")
        
        context = sample[input_key[args.dataset]]
        query = ""

        try:
            # Extract top K attributed sentences by ContextCiter
            if "Qwen3" in args.model_name:
                cc = Qwen3ContextCiter(model, tokenizer, context, query, num_ablations=args.num_ablations)
            else:
                cc = ContextCiter(model, tokenizer, context, query, num_ablations=args.num_ablations)
            cc.prompt_template = get_prompt_template(args.dataset)
            cc.generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "use_cache": True,
            }

            if "Llama-3" in args.model_name:
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                cc.generate_kwargs["eos_token_id"] = terminators

            results = cc.get_attributions(as_dataframe=True, top_k=args.num_sents, verbose=False)
            df = results.data
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory during generation for sample {idx}. Skipping...")
            torch.cuda.empty_cache()
            skipped_samples.append(sample)
            continue

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
    
    if skipped_samples:
        skipped_save_path = args.save_path.replace('.json', '_skipped.json')
        with open(skipped_save_path, 'w') as fh:
            json.dump(skipped_samples, fh, indent=4)

if __name__ == "__main__":
    main()
    