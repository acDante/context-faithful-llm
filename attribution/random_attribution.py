# Extract K attributed sentences randomly
import random
import json
import argparse
from tqdm import tqdm
import copy
from typing import List, Tuple
import nltk
import os
from dotenv import load_dotenv
from datasets import load_dataset


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

def extract_random_sentences(document: str, n: int) -> List[str]:
    sentences, _ = split_into_sentences(document)
    
    # Handle edge cases
    if n <= 0 or not sentences:
        return []
    
    # If n exceeds available sentences, return all sentences
    if n >= len(sentences):
        return sentences
    
    # Randomly select n indices and sort to maintain original order
    selected_indices = sorted(random.sample(range(len(sentences)), n))
    
    # Return sentences in original document order
    return [sentences[i] for i in selected_indices]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn', 'ccsum', 'summscreen', 'gov_report', 'qmsum'])
    parser.add_argument("--num_samples", default=1000, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of random sentences to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    load_dotenv("../.env")

    test_data = load_data(args.dataset)
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))

    processed_samples = []
    for idx, sample in tqdm(enumerate(test_data)):
        if idx % 100 == 0:
            print(f"Currently processing: {idx}-th sample.\n") 
        
        document = sample[input_key[args.dataset]]
        random_sents = extract_random_sentences(document, n=args.num_sents)

        attributed_sents = []
        for sent in random_sents:
            attributed_sents.append(
                {
                    "input_sequence": sent,
                    "score": 1.0
                }
            )
        
        processed_sample = copy.deepcopy(sample)
        processed_sample.update({"attributed_sents": attributed_sents})
        processed_samples.append(processed_sample)

    # Save the processed instances to a JSON file
    with open(args.save_path, 'w') as fh:
        json.dump(processed_samples, fh, indent=4)


if __name__ == "__main__":
    main()