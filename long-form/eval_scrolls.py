import argparse
import json
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summaries on long-form datasets using vLLM")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", 
                        help="Model name")
    parser.add_argument("--dataset", choices=["qmsum", "summscreen"], default="qmsum",
                        help="Dataset used for evaluation")
    parser.add_argument("--save-path", default="results", help="Path to save the predictions")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=34000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attr_data_path", type=str, default=None, help="Path to the processed data with annotated attributions")
    parser.add_argument("--method", type=str, default="base", choices=['base', 'base+impt'], help="which attribution-guided generation approach to use")

    return parser.parse_args()

def get_prompt(doc, important_sents, args):
    if args.method == "base": 
        prompt_template = {
            "qmsum": "Read the following meeting transcript. Produce a summary in 5 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{}",
            "summscreen": "Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{}"
        }

    elif args.method == "base+impt":
        prompt_template = {
            "qmsum": "Read the following meeting transcript. Produce a summary in 5 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{}",
            "summscreen": "Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{}"
        }
        for dataset_name in prompt_template.keys():
            prompt_template[dataset_name] += "\nYou should only focus on the following key points:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(important_sents)]) + "\n"

    return prompt_template[args.dataset].format(doc)

def process_in_batches(data, batch_size):
    """Split data into batches"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

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

def extract_summary(text):
    """Extract summary from structured output format."""
    if "Summary:" in text:
        # Find the start of "Summary:" and extract everything after it
        start = text.find("Summary:") + len("Summary:")
        summary = text[start:].strip()
        
        # Remove any trailing formatting or extra text after the summary
        # Split by common delimiters and take the first part
        for delimiter in ['\n\n', '\n---', '\nNote:', '\nAdditional']:
            if delimiter in summary:
                summary = summary.split(delimiter)[0]
                break
        
        return summary.strip()
    return text.strip()

def main():
    args = parse_args()

    # Initialize the model
    llm = LLM(
        model = args.model,
        max_model_len = args.max_model_len,
        gpu_memory_utilization = args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,  # Reduce if OOM, increase for better throughput
        swap_space=4,   # GB of CPU memory for overflow
        enforce_eager=False,  # Keep as False for better performance
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

    # # Prompt template (adapted from longform-chat)
    # prompt_template = {
    #     "qmsum": "Read the following meeting transcript. Produce a summary in 5 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{}",
    #     "summscreen": "Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{}"
    # }

    # # Create chat prompts
    # chat_messages = [
    #     [
    #         {"role": "system", "content": "You are an expert summarization assistant."},
    #         {"role": "user", "content": prompt_template[args.dataset].format(doc)}
    #     ]
    #     for doc in documents
    # ]

    # Create chat prompts
    chat_messages = []
    for item in data:
        doc = item['input']
        attributed_sents = [sent['input_sequence'] for sent in item['attributed_sents']]
        prompt = get_prompt(doc, attributed_sents, args)
        chat_messages.append([
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt}
        ])


    # Generate summaries
    predictions = []
    outputs = llm.chat(messages=chat_messages,
                       sampling_params=sampling_params,
                       use_tqdm=True)
                    #    chat_template_kwargs={"enable_thinking": False},  # Disable thinking for Qwen3 models
                    #    use_tqdm=True)
    
    for output in outputs:
        prediction = extract_summary(output.outputs[0].text)
        predictions.append(prediction)

    # Save the predictions to a local JSON file
    results = []
    for i, (item, prediction) in enumerate(zip(data, predictions)):
        result_item = dict(item)  # Copy all original fields
        result_item['generated_summary'] = prediction  # Add generated summary
        results.append(result_item)
    
    short_model_name = get_short_model_name(args.model)
    filename = os.path.join(args.save_path, f"{short_model_name}_{args.dataset}_{args.split}_{len(results)}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Generated {len(results)} summaries → {filename}")

if __name__ == "__main__":
    main()

