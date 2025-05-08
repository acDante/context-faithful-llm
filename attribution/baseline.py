# Implement lead-k (first k sentences) and random-k baseline (random k sentences)
import json
import argparse
from tqdm import tqdm
import random
import copy
import numpy as np
from typing import List, Tuple, Dict
import nltk
from datasets import load_dataset


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
    parser.add_argument("--num_samples", default=1000, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of most important sentences to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")
    parser.add_argument("--method", default='lead3', type=str, choices=['lead3', 'random'], help="How to extract important sentences")

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

def main():

    args = parse_args()
    test_data = load_data(args.dataset)
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    processed_samples = []

    for idx, sample in tqdm(enumerate(test_data)):
        if idx % 100 == 0:
            print((f"Currently processing: {idx}-th sample.\n"))
        
        doc = sample[input_key[args.dataset]]
        sentences, separators = split_into_sentences(doc)
        if args.method == "lead3":
            selected_sentences = sentences[:min(args.num_sents, len(sentences))]
        elif args.method == "random":
            selected_sentences = random.sample(sentences, min(args.num_sents, len(sentences)))
        
        attributed_sents = []
        for sent in selected_sentences:
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
        processed_samples.append(processed_sample)

    # Save the processed instances to JSON file
    with open(args.save_path, 'w') as fh:
        json.dump(processed_samples, fh, indent=4)


if __name__ == "__main__":
    main()