import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import copy

from cad import CAD


input_key = {
    "xsum": "document",
    "cnn_dm": "article"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights"
}

def get_prompt(doc, important_sents, schema, dataset):
    prompt = ""
    if schema == "base":
        if dataset == "xsum":
            instruction = "Summarise the document below in one sentence:"
        elif dataset == "cnn_dm":
            instruction = "Summarise the document below:"

        prompt = f"{instruction}\n{doc}"
    
    elif schema == "base+impt":
        if dataset == "xsum":
            instruction = "Summarise the document below in one sentence:"
        elif dataset == "cnn_dm":
            instruction = "Summarise the document below:"
        
        prompt = f"{instruction}\n{doc}\nYou should pay attention to the following main points:\n"  # TODO: try subtracting these sentences from the input doc?
        for id, sent in enumerate(important_sents):
            prompt += f"{str(id + 1)}. {sent}\n"  # TODO: try list important sentences according to their original order
    
    elif schema == "impt_only":
        if dataset == "xsum":
            prompt = f"Summarise the following points in one sentence:\n"
        elif dataset == "cnn_dm":
            prompt = f"Summarise the following points by sentences:\n"
        
        for id, sent in enumerate(important_sents):
            prompt += f"{str(id + 1)}. {sent}\n"

    return prompt

def get_question_prompt(doc, schema, dataset):
    prompt = ""
    if schema == "base":
        prompt = "Summary:"

    elif schema == "base+impt":
        if dataset == "xsum":
            instruction = "Summarise the document below in one sentence:"
        elif dataset == "cnn_dm":
            instruction = "Summarise the document below:"
        
        prompt = f"{instruction}\n{doc}"
    
    return prompt

def load_data(args):
    # Load test data
    if args.attr_data_path:
        with open(args.attr_data_path, "r") as fin:
            test_data = json.load(fin)
            test_data = test_data[:args.num_samples]
    else:
        if args.dataset == "xsum":
            test_data = load_dataset("xsum", split="test")
        elif args.dataset == "cnn_dm":
            test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
        
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    return test_data

# Generate the output text given the model and input ids
def generate(model, tokenizer, input_ids, model_name, max_new_tokens=128):
    if "Mistral" in model_name:
        output_ids = model.generate(input_ids,
                                    do_sample=False,
                                    max_new_tokens=max_new_tokens,
                                    temperature=0.0)
    elif "Llama-3" in model_name:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output_ids = model.generate(input_ids,
                                    do_sample=False,
                                    max_new_tokens=max_new_tokens,
                                    temperature=0.0,
                                    eos_token_id=terminators)
    
    output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
    return output_text

# Generate the output text using CAD model
def cad_generate(model, tokenizer, question_prompt, prompt, model_name, alpha, max_new_tokens=128):
    if "Mistral" in model_name:
        raw_output = model.generate(texts=question_prompt,
                                    texts_with_context=prompt,
                                    alpha=alpha,
                                    do_sample=False,
                                    top_k=None,
                                    top_p=None,
                                    max_new_tokens=max_new_tokens,
                                    temperature=0.0)
    
    elif "Llama-3" in model_name:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        raw_output = model.generate(texts=question_prompt,
                                    texts_with_context=prompt,
                                    alpha=alpha,
                                    do_sample=False,
                                    top_k=None,
                                    top_p=None,
                                    max_new_tokens=max_new_tokens,
                                    temperature=0.0,
                                    eos_token_id=terminators)
    
    return raw_output[0]

# Post process the generated output text
def post_process(output_text, dataset):
    if dataset == "xsum":
        output_text = output_text.split('.')[0] + "."
        if "\n\n" in output_text:
            output_text = output_text.split("\n\n")[-1]
    
    elif dataset == "cnn_dm":
        output_text = output_text

    return output_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dataset", default="xsum", type=str, help="name of the dataset to evaluate on")
    parser.add_argument("--attr_data_path", type=str, default=None, help="Path to the processed test data with extracted important sentences")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test samples to evaluate on")
    parser.add_argument("--log_path", default="results/summary", type=str)
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--schema", default="base", type=str, choices=['base', 'base+impt', 'impt_only'],
                        help="whether to include important sentences in the prompt")
    parser.add_argument("--use_cad", action="store_true", help="Use context-aware decoding")
    parser.add_argument("--alpha", default=0.5, type=float, help="Parameter for context-aware decoding")
    
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    login("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")

    # Load test dataset
    test_data = load_data(args)
    
    # Load model and tokenizer
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    context_window_length = getattr(config, 'max_position_embeddings', 
                                    getattr(config, 'n_positions', None))

    base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      torch_dtype=torch.bfloat16,
                                                      device_map="auto",
                                                      use_auth_token=True,
                                                      cache_dir="/mnt/ssd/llms")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side="left")
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    tokenizer.model_max_length = context_window_length

    # Initialise the model for constrastive decoding
    model = base_model
    if args.use_cad:
        model = CAD(model=base_model, tokenizer=tokenizer)

    log_path = Path(args.log_path)
    output_path = log_path / f"{args.exp_name}_preds.json"

    processed_samples = []
    for idx, sample in tqdm(enumerate(test_data)):
        # TODO: deal with the case when attribution is None (skip those data)
        doc = sample[input_key[args.dataset]]
        # doc = sample['document']
        # summary = sample['summary']

        if "attributed_sents" in sample.keys():
            if sample['attributed_sents'] == None:
                continue
            else:
                attributed_sents = [sent['input_sequence'] for sent in sample['attributed_sents']]
        else:
            attributed_sents = None
        
        prompt = get_prompt(doc, attributed_sents, args.schema, args.dataset)
        messages = [{
            "role": "user", 
            "content": prompt
        }]
        
        if args.use_cad:
            # TODO: with v.s. without chat template?
            question_prompt = get_question_prompt(doc, args.schema, args.dataset)
            question_messages = [{
                "role": "user",
                "content": question_prompt
            }]

            # Add chat templates
            question_prompt_text = tokenizer.apply_chat_template(question_messages,
                                                                 tokenize=False,
                                                                 add_generation_prompt=True)
            prompt_text = tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=True)
            output_text = cad_generate(model, tokenizer, 
                                       question_prompt=question_prompt_text, prompt=prompt_text, 
                                       model_name=model_name, alpha=args.alpha, max_new_tokens=128)
        
        else:
            input_ids = tokenizer.apply_chat_template(messages, 
                                                      return_tensors="pt", 
                                                      add_generation_prompt=True).to(model.device)
            
            output_text = generate(model, tokenizer, input_ids, model_name, max_new_tokens=128)
        
        output_text = post_process(output_text, args.dataset)
        # output_text = output_text.split('.')[0] + "."  # Note: we only keep the first sentence (for testing on XSum); for general summarisaiton task: keep all the content before \n\n or until the last complete sentence [TODO]
        processed_sample = copy.deepcopy(sample)
        # Save the generated summary
        processed_sample.update({"generated_summary": output_text})
        processed_samples.append(processed_sample)
    
    with open(output_path, "w") as fh:
        json.dump(processed_samples, fh, indent=4)


if __name__ == '__main__':
    main()