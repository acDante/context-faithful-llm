# Extract top k important sentences from the input document based on attribution scores

import inseq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import evaluate

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import copy


# Check if the current token is the end of a sentence
# Note that: this algo cannot handle the corner case with abbreviation, e.g. "P.E."
def is_sentence_ending(text):
    if text.endswith(("!", ".", "?")):
        return True
    if text.endswith((".\"", "?\"", "!\"")):
        return True
    
def aggregate_by_sents(attr_output, doc_start_pos):
    # Aggregate the attribution scores for each sentence (treating the instruciton and document separately)
    ends = [i + 1 for i, t in enumerate(attr_output[0].source) if is_sentence_ending(t.token)] + [len(attr_output[0].source) - 1]
    starts = [doc_start_pos] + [i + 1 for i, t in enumerate(attr_output[0].source) if is_sentence_ending(t.token)]
    source_spans = list(zip(starts, ends))

    # Remove empty spans 
    processed_spans = [(0, doc_start_pos)]
    for span in source_spans:
        if span[0] + 1 < span[1]:
            processed_spans.append(span)

    res = attr_output.aggregate("spans", source_spans=processed_spans)
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="extra_cnn", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn'])
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--attribution", default="attention", type=str, help="Type of attribution method to use")
    parser.add_argument("--aggregate_func", default="max", choices=["max", "mean"], help="How to aggregate the attribution scores across all generated tokens")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of most important sentences to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    access_token = "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"

    if args.dataset == "extra_cnn":
        test_data = datasets.load_dataset("eReverter/cnn_dailymail_extractive", split="test")
    elif args.dataset == "cnn_dm":
        test_data = datasets.load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif args.dataset == 'xsum':
        test_data = datasets.load_dataset("xsum", split="test")
    
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    # Use the same prompt from extractive_sum.py
    instruction = "Summarise the document below:"

    model = inseq.load_model(args.model_name, args.attribution)
    # Get the start position of the tokenized document for aggregating attribution scores
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              token=access_token)
    encoded_instruct = tokenizer(instruction, return_tensors="pt", add_special_tokens=False).input_ids
    doc_start_pos = len(encoded_instruct[0])

    processed_samples = []
    for idx, sample in tqdm(enumerate(test_data)):
        article = sample['src']
        doc = " ".join(article)
        prompt = f"{instruction}\n\n{doc}\n\nSummary:"

        # Get the attribution scores for each token in the input document
        out = model.attribute(
            prompt,
            n_steps=100,
            generation_args={"max_new_tokens": 128,
                             "do_sample": False,
                             "temperature": 0.0},
            internal_batch_size=50
        )

        # Aggregate the attribution scores for each input sentence
        res = aggregate_by_sents(out, doc_start_pos)

        # Aggregate the scores for each sentence w.r.t. all the generated tokens
        score_dict = res.get_scores_dicts()
        src_attr_dict = score_dict[0]['source_attributions']
        input_sent_scores = defaultdict(list)
        for output_tok in src_attr_dict.keys():
            for input_sent in src_attr_dict[output_tok].keys():
                cleaned_sent = input_sent[1].replace("▁", " ").strip()
                if cleaned_sent == "</s>" or cleaned_sent == instruction or cleaned_sent == "Summary:":  # ignore the last two sents
                    continue
                input_sent_scores[cleaned_sent].append(src_attr_dict[output_tok][input_sent])
        
        # Aggregate the input attribution scores based on the chosen method
        aggregated_sent_scores = {}
        if args.aggregate_func == "max":
            for sent in input_sent_scores.keys():
                aggregated_sent_scores[sent] = max(input_sent_scores[sent])
        
        elif args.aggregate_func == "mean":
            for sent in input_sent_scores.keys():
                aggregated_sent_scores[sent] = sum(input_sent_scores[sent]) / len(input_sent_scores[sent])
        
        # Extract top K important sentences
        sorted_sent_scores = dict(sorted(aggregated_sent_scores.items(), key=lambda x: x[1], reverse=True))
        top_k_sents = list(sorted_sent_scores.keys())[:args.num_sents]

        # Save processed instances with the top K important sentences
        processed_sample = copy.deepcopy(sample)
        processed_sample.update({"important_sents": top_k_sents})
        processed_samples.append(processed_sample)

        # For extractive CNN data: save the gold extractive summary in a readable format
        if args.dataset == "extra_cnn":
            label = sample['labels']
            annotated_sents = [article[idx] for idx in range(len(label)) if label[idx]== 1]
            processed_sample.update({"extractive_summary": annotated_sents})

    # Save the processed instances to a JSON file
    save_path = Path(args.save_path)
    with open(save_path, 'w') as fh:
        json.dump(processed_samples, fh, indent=4)


if __name__ == "__main__":
    main()
