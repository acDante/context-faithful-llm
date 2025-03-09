"""Use language model to diretly generate attributed sentences"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import evaluate

import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    access_token = "HF_TOKEN"
    # Test with extractive CNN/DM data
    test_data = datasets.load_dataset("eReverter/cnn_dailymail_extractive", split="test")
    test_data = test_data[0]
    # print(len(test_data['src']))

    # Append index to each sentence in the document
    article = ""
    for idx, sent in enumerate(test_data['src']):
        article += f"{str(idx + 1)}. {sent}\n"
    print(article)

    prompt = "Extract the top three most important sentences and only output their indices:\n" + article
    # prompt = "Find the top three most important sentences in the article and only output their indices:\n" + article
    print(prompt)

    # model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16,
                                                 token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=True,
                                              padding_side="left",
                                              token=access_token)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    messages = [    
        # {
        #     "role": "system",
        #     "content": "You are a helpful assistant."
        # },
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, truncation=False, return_tensors="pt")
    input_ids = inputs.to("cuda")

    outputs = model.generate(input_ids,
                             do_sample=False,
                             max_new_tokens=200,
                             temperature=0.75,
                             top_p=0.9)
    raw_output = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
    output = raw_output

    print(output)

if __name__ == "__main__":
    main()