import json
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use for summarization")
    parser.add_argument("--dataset", type=str, default="xsum", 
                        choices=["xsum", "cnn_dailymail"],
                        help="Name of summarisation dataset")
    parser.add_argument("--test_size", type=int, default=3000, 
                        help="Number of test examples to use for evaluation")
    parser.add_argument("--method", type=str, default="base", choices=["base", "attr"])
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the prediction files")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    return args

def new_extract_summary_and_attributions(text):
    """Extract summary and attributions from model output."""
    summary = ""
    attributions = []
    cleaned_attributions = []
    
    # Find summary
    if "Summary:" in text:
        summary_start = text.find("Summary:") + len("Summary:")
        summary_end = text.find("Attributions:") if "Attributions:" in text else len(text)
        summary = text[summary_start:summary_end].strip()
    
    # Find attributions if present
    if "Attributions:" in text:
        attr_start = text.find("Attributions:") + len("Attributions:")
        attr_text = text[attr_start:].strip()
        
        # Split by numbered items
        import re
        attr_items = re.split(r'\d+\.', attr_text)
        attributions = [item.strip() for item in attr_items if item.strip()]
    
        cleaned_attributions = []
        for sent in attributions:
            cleaned_sent = sent
            if sent.startswith('"') and sent.endswith('"'):
                cleaned_sent = cleaned_sent[1:-1]
            cleaned_attributions.append(cleaned_sent)

    return summary, cleaned_attributions

def extract_summary_and_attributions(text):
    """Split summary and attributed sentences from the raw output"""
    summary = ""
    attributions = []
    cleaned_attributions = []
    parts = text.split("Summary:")
    if len(parts) > 1:
        # Extract summary - everything between "Summary:" and "Attributions:"
        summary_section = parts[1].split("Attributions:")[0].strip()
        summary = summary_section
        
        # Extract attributions section - everything after "Attributions:"
        if "Attributions:" in parts[1]:
            attributions_section = parts[1].split("Attributions:")[1].strip()
            
            # Process the attributions using regex to handle line breaks
            import re
            attribution_items = re.findall(r'\d+\.\s+(.*?)(?=\d+\.\s+|\Z)', attributions_section + "\n", re.DOTALL)
            
            # Clean up each attribution by removing quotations
            attributions = [item.strip() for item in attribution_items if item.strip()]

            for sent in attributions:
                cleaned_sent = sent
                if sent.startswith('"') and sent.endswith('"'):
                    cleaned_sent = cleaned_sent[1:-1]
                cleaned_attributions.append(cleaned_sent)
        
    return summary, cleaned_attributions

def create_prompt(document, method="base"):
    """Create prompt for different experiment settings"""
    # Baseline: only generate the summary
    if method == "base":
        prompt_template = """
        Summarize the following news article in a single sentence.

        Article:
        {article}

        Provide your response in the following format:
        Summary: [Your one-sentence summary] 
        """
    
    # Generate LLM Attribution: generate the summary and provide supporting evidence
    elif method == "attr":
        prompt_template = """
        You are tasked with summarizing the following news article in a single sentence. After providing the summary, identify 1-4 key sentences from the original document that support your summary.

        Article:
        {article}

        Provide your response in the following format:
        Summary: [Your one-sentence summary]

        Attributions:
        1. the first evidence sentence from input document
        [Include 2-3 additional evidence sentences in the same format as above only if you use the information to generate the summary.]
        """
    
    return prompt_template.format(article=document)

def get_dataset_config(dataset_name):
    dataset_configs = {
        "xsum": {
            "name": "xsum",
            "document_field": "document",
            "summary_field": "summary",
            "id_field": "id"
        },
        "cnn_dailymail": {
            "name": "cnn_dailymail",
            "version": "3.0.0",
            "document_field": "article",
            "summary_field": "highlights",
            "id_field": "id"
        },
        # Add more datasets as needed
    }

    configs = dataset_configs.get(dataset_name, None)
    return configs


def main():

    args = parse_args()

    # Prepare output diretory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)

    # Set output path
    model_short_name = args.model_name.split("/")[-1].lower()
    output_file = f"{args.dataset}-{model_short_name}-{args.method}.json"
    output_path = os.path.join(args.output_dir, output_file)

    # Load summarisation dataset
    split_setting = f"test[:{args.test_size}]" if args.test_size > 0 else "test"
    if "version" in dataset_config:
        dataset = load_dataset(dataset_config["name"], dataset_config["version"], split=split_setting)
    else:
        dataset = load_dataset(dataset_config["name"], split=split_setting)

    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset.")
    # xsum_dataset = load_dataset("xsum", split="test[:3000]")
    # print(len(xsum_dataset))

    # Load Qwen model and tokenizer
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_predictions = []

    print(f"Processing {len(dataset)} samples from {args.dataset}...")

    for i, example in enumerate(tqdm(dataset)):
        document = example[dataset_config["document_field"]]
        reference = example[dataset_config["summary_field"]]

        prompt = create_prompt(document, method=args.method)
    
        messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise, factual summaries of documents."},
                {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        summary, attributions = new_extract_summary_and_attributions(response)

        result = {
            "id": example.get(dataset_config["id_field"], i),
            "document": document,
            "reference_summary": reference,
            "generated_summary": summary,
            "attributions": attributions,
            "raw_output": response
        }

        all_predictions.append(result)

        # Save results periodically (every 1000 examples)
        if (i + 1) % 1000 == 0:
            with open(output_path, "w") as f:
                json.dump(all_predictions, f, indent=4)
            print(f"Saved intermediate results after {i + 1} examples")

    # Save final predictions to file
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=4)

    print(f"Saved results to {output_path}.")

if __name__ == '__main__':
    main()