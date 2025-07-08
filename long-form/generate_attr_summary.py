# Script for extracting generative attribution with LLMs

import argparse
import json
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract generative attribution on long-form datasets using vLLM")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", 
                        help="Model name")
    parser.add_argument("--dataset", choices=["qmsum", "summscreen"], default="summscreen",
                        help="Dataset used for evaluation")
    parser.add_argument("--save-path", default="results/attribution", help="Path to save the predictions")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=34000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num_sents", default=3, type=int, help="Number of attributed sentences to extract")
    parser.add_argument("--attr_type", default="sentence", type=str, choices=["sentence", "fact"])

    return parser.parse_args()

def get_short_model_name(model_name):
    model_mapping = {
        "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
        "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
        "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",
        "Qwen/Qwen3-8B": "qwen3-8b",
        "Qwen/Qwen3-32B": "qwen3-32b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
        "Qwen/Qwen2.5-14B-Instruct": "qwen2.5-14b",
    }

    return model_mapping.get(model_name, model_name.replace("/", "_"))

def extract_sentences_and_summary(text):
    """
    Extract key sentences and summary from LLM output.
    
    Args:
        text (str): The LLM output text containing key sentences and summary
        
    Returns:
        tuple: (list of key sentences, summary string)
    """
    try:
        # Find the summary section
        summary_start = text.find("Summary:")
        
        if summary_start == -1:
            return [], ""
        
        # Extract everything before "Summary:" as key sentences text
        key_sentences_text = text[:summary_start].strip("\n[]")
        
        # Extract the summary
        summary = text[summary_start + len("Summary:"):].strip()
        
        # Parse the key sentences into a list
        sentences = []
        
        # Split the text into lines
        lines = key_sentences_text.split('\n')
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number followed by period (e.g., "1.")
            if re.match(r'^\d+\.', line):
                # Remove the number prefix
                sentence = re.sub(r'^\d+\.\s*', '', line).strip()
                if sentence:
                    sentences.append(sentence)
            # Also check for bullet points for backward compatibility
            elif line.startswith('- ') or line.startswith('* '):
                sentence = line[2:].strip()
                if sentence:
                    sentences.append(sentence)
        
        return sentences, summary
    
    except Exception as e:
        print(f"Error parsing text: {e}")
        return [], ""
    
def main():
    args = parse_args()

    # Initialize the model
    llm = LLM(
        model = args.model,
        max_model_len = args.max_model_len,
        gpu_memory_utilization = args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,  # Reduce if OOM, increase for better throughput
        swap_space=4,   # GB of CPU memory for overflow
        enforce_eager=False,  # Keep as False for better performance
        download_dir="/mnt/ceph_rbd/llms"
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Load the dataset
    dataset_map = {"qmsum": "qmsum", "summscreen": "summ_screen_fd"}
    dataset = load_dataset("tau/scrolls", dataset_map[args.dataset])[args.split]
    data = dataset.select(range(min(args.max_samples, len(dataset))))

    documents = [item['input'] for item in data]
    references = [item['output'] for item in data]

    # Prompt template (adapted from longform-chat)
    if args.attr_type == "fact":
        prompt_template = {
            "qmsum": "Read the following meeting transcript. Extract a list of {num_sents} key decisions, action items and discussion points from the input document and then produce a summary in 5 sentences only focusing on the extracted facts. You must give your answer in a structured format: \"Key Facts:\n1. sentence1, 2. sentence2, ...\nSummary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{doc}",
            "summscreen": "Read the following TV episode transcript. Extract a list of {num_sents} key plot developments and story events from the input document and then produce a summary in 5 sentences only focusing on the extracted facts. You must give your answer in a structured format: \"Key Facts:\n1. sentence1, 2. sentence2, ...\nSummary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{doc}"
        }

    elif args.attr_type == "sentence":
        prompt_template = {
            "qmsum": "Read the following meeting transcript. Extract a list of {num_sents} key sentences from the input document and then produce a summary in 5 sentences only focusing on the extracted sentences. You must give your answer in a structured format: \"Key Sentences:\n1. sentence1, 2. sentence2, ...\nSummary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{doc}",
            "summscreen": "Read the following TV episode transcript. Extract a list of {num_sents} key sentences from the input document and then produce a summary in 5 sentences only focusing on the extracted sentences. You must give your answer in a structured format: \"Key Sentences:\n1. sentence1, 2. sentence2, ...\nSummary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{doc}"
        }

    # Create chat prompts
    chat_messages = [
        [
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt_template[args.dataset].format(doc=doc, num_sents=args.num_sents)}
        ]
        for doc in documents
    ]

    # Generate summaries with attributed sentences
    predictions = []
    if "Qwen" in args.model:
        outputs = llm.chat(messages=chat_messages,
                        sampling_params=sampling_params,
                        use_tqdm=True,
                        chat_template_kwargs={"enable_thinking": False})
        
    else:
        outputs = llm.chat(messages=chat_messages,
                           sampling_params=sampling_params,
                           use_tqdm=True)

    raw_outputs = []
    for output in outputs:
        raw_outputs.append(output.outputs[0].text)
    
    # Extract key sentences and summaries from the raw output and save to local JSON file
    results = []
    for i, (item, prediction) in enumerate(zip(data, raw_outputs)):
        key_sentences, summary = extract_sentences_and_summary(prediction)
        attributed_sentts = []
        for sent in key_sentences:
            sent = sent.strip()
            if len(sent) > 0:
                attributed_sentts.append(
                    {
                        "input_sequence": sent,
                        "score": 1.0,  # Assuming all sentences are equally important
                    }
                )
        result_item = dict(item)
        result_item['attributed_sents'] = attributed_sentts
        result_item['generated_summary'] = summary
        result_item['raw_output'] = prediction
        results.append(result_item)
    
    short_model_name = get_short_model_name(args.model)
    filename = os.path.join(args.save_path, f"{short_model_name}_{args.dataset}_{args.split}_{len(results)}_gen_attr_{args.attr_type}_num{args.num_sents}.json")

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Generated {len(results)} summaries → {filename}")

if __name__ == "__main__":
    main()
