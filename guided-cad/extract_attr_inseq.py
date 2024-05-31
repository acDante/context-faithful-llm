# Extract top k important sentences from the input document based on attribution scores

import inseq
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
import evaluate

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import copy


input_key = {
    "xsum": "document",
    "cnn_dm": "article"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights"
}

# Check if the current token is the end of a sentence
# Note that: this algo cannot handle the corner case with abbreviation, e.g. "P.E."
def is_sentence_ending(text):
    if text.endswith(("!", ".", "?")):
        return True
    if text.endswith((".\"", "?\"", "!\"")):
        return True

# Get the number of encoded tokens for a given text
def get_token_length(text, tokenizer):
    encoded_text = tokenizer(text, 
                             return_tensors="pt", 
                             add_special_tokens=False).input_ids
    
    return encoded_text.shape[-1]

# Remove special characters from inseq
def clean_token(token):
    processed_token = token.replace("▁", " ")  
    return processed_token

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xsum", type=str, choices=['cnn_dm', 'xsum', 'extra_cnn'])
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--attribution", default="attention", type=str, help="Type of attribution method to use")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test instances to processs")
    parser.add_argument("--num_sents", default=3, type=int, help="Number of most important sentences to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the processed instances with the most important sentences")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    login("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")

    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # Load the test data
    model_name = args.model_name
    if args.dataset == "extra_cnn":
        test_data = load_dataset("eReverter/cnn_dailymail_extractive", split="test")
    elif args.dataset == "cnn_dm":
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif args.dataset == 'xsum':
        test_data = load_dataset("xsum", split="test")
    
    test_data = test_data.select(range(min(args.num_samples, len(test_data))))

    ### Load model and tokenizer
    # model_name = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(model_name)
    context_window_length = getattr(config, 'max_position_embeddings', 
                                    getattr(config, 'n_positions', None))

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 use_auth_token=True,
                                                 cache_dir="/mnt/ssd/llms")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = context_window_length

    ### Compute the attribution for each instance
    processed_samples = []
    save_path = Path(args.save_path)
    with open(save_path, "w"):
        pass

    for idx, sample in tqdm(enumerate(test_data)):
            
        if args.dataset == "extra_cnn":
            article = sample['src']
            doc = " ".join(article)
        else:
            doc = sample[input_key[args.dataset]]

        instruction = "Summarise the document below:"  # TODO: this prompt does not work well for Llama2-chat model
        prompt_message = f"{instruction}\n{doc}"  # Note: adding a newline character can change the attribution distribution
        messages = [{
            "role": "user", 
            "content": prompt_message
        }]

        prompt = tokenizer.apply_chat_template(messages, 
                                               return_tensors="pt", 
                                               add_generation_prompt=True).to(model.device)
        # Note: long context (>2500 tokens) will cause CUDA out of memory issue when running model.attribute()
        # Here we skip those long instances
        if prompt.shape[-1] > 2000:
            processed_sample = copy.deepcopy(sample)
            processed_sample.update({"attributed_sents": None})
            processed_sample.update({"generated_summary": output_text})
            processed_samples.append(processed_sample)
            continue

        prompt_text = tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)

        inseq_model = inseq.load_model(model, args.attribution, tokenizer=model_name)
        output_ids = model.generate(prompt,
                                    do_sample=False,
                                    max_new_tokens=64,
                                    temperature=0.0)

        output_text = tokenizer.decode(output_ids[0, prompt.shape[1]:], skip_special_tokens=False)
        output_text = output_text.split('.')[0] + "."  # Note: we only keep the first sentence (for testing on XSum); for general summarisaiton task: keep all the content before \n\n or until the last complete sentence [TODO]
        # output_text = tokenizer.decode(output_ids[0, prompt.shape[1]:], skip_special_tokens=True)

        # print(output_text)
        out = inseq_model.attribute(
            input_texts=prompt_text,
            generated_texts=prompt_text + output_text,
        )

        # out.show()

        ### Aggregate the attribution scores for each input sentence
        # Process intrucitons and special tokens in chat template separately
        start_marker = "<s><s>[INST]"
        end_marker = "[/INST]"

        # Calculate the token length for each part of the prompt
        len_start_marker = get_token_length(start_marker, tokenizer)
        len_end_marker = get_token_length(end_marker, tokenizer)
        len_instruction = get_token_length(instruction, tokenizer)
        len_prompt = get_token_length(prompt_message, tokenizer)
        total_prompt_len = len_start_marker + len_prompt

        doc_start_pos = len_start_marker + len_instruction
        start_span = (0, len_start_marker)
        instr_span = (len_start_marker, len_start_marker + len_instruction)
        end_span = (total_prompt_len, total_prompt_len + len_end_marker)

        ends = [i + 1 for i, t in enumerate(out[0].target) if is_sentence_ending(t.token) and i < total_prompt_len] + [total_prompt_len]
        starts = [doc_start_pos] + [i + 1 for i, t in enumerate(out[0].target) if is_sentence_ending(t.token) and i < total_prompt_len]
        spans = [start_span, instr_span] + list(zip(starts, ends)) + [end_span]

        # Remove invalid spans 
        processed_spans = []
        for span in spans:
            if span[0] + 1 < span[1]:
                processed_spans.append(span)
        
        # print(processed_spans)
        res = out.aggregate("spans", target_spans=processed_spans)
        # res.show()

        ### Assign a single attribution score to each input sentence and extract top K important sentences
        tok_out = res.aggregate()
        prompt_last_index = tok_out[0].attr_pos_start

        input_sequences = [clean_token(t.token) for t in tok_out[0].target[2:prompt_last_index-1]]  # Note: ignore the special tokens and instruction prompt
        cleaned_sequences = []
        for seq in input_sequences:
            processed_seq = seq.replace("<0x0A>", " ").strip()
            cleaned_sequences.append(processed_seq)

        attr_scores = tok_out[0].target_attributions[2:prompt_last_index-1].tolist()
        assert(len(cleaned_sequences) == len(attr_scores))

        # Note: we only consider the maximum attribution score for each sentence [TODO: other aggregation method?]
        sent_scores = dict()
        for seq_ix, seq in enumerate(cleaned_sequences):
            sent_scores[seq] = max(attr_scores[seq_ix])

        # Extract top K important sentences
        sorted_sent_scores = dict(sorted(sent_scores.items(), key=lambda x: x[1], reverse=True))
        top_k_sents = list(sorted_sent_scores.keys())[:args.num_sents]

        attributed_sents = []
        for sent in top_k_sents:
            attributed_sents.append(
                {
                    "input_sequence": sent,
                    "score": sent_scores[sent]
                }
            )  # TODO: also store the index of sentences in the document? 
        
        processed_sample = copy.deepcopy(sample)
        # For extractive CNN data: save the gold extractive summary in a readable format
        if args.dataset == "extra_cnn":
            label = sample['labels']
            annotated_sents = [article[idx] for idx in range(len(label)) if label[idx]== 1]
            processed_sample.update({"gold_extractive_summary": annotated_sents})
        
        # Save the top K important sentences and their attribution scores
        processed_sample.update({"attributed_sents": attributed_sents})

        # Save the generated summary
        processed_sample.update({"generated_summary": output_text})
        processed_samples.append(processed_sample)

        if idx % 100 == 0:
            print(f"Currently processing: {idx}-th sample")
            with open(args.save_path, 'a') as fh:
                json.dump(processed_samples, fh, indent=4)
            processed_samples = []
    
    # Save the processed instances to a JSON file
    # save_path = Path(args.save_path)
    # with open(save_path, 'w') as fh:
    #     json.dump(processed_samples, fh, indent=4)


if __name__ == "__main__":
    main()