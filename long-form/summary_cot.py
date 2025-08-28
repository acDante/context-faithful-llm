import argparse
import json
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Summary Chain of Thought using vLLM")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", 
                        help="Model name")
    parser.add_argument("--dataset", choices=["qmsum", "summscreen", "gov_report"], default="qmsum",
                        help="Dataset used for evaluation")
    parser.add_argument("--save-path", default="results/summary", help="Path to save the predictions")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens-qa", type=int, default=2048, help="Max number of tokens for generating QA-based plan")
    parser.add_argument("--max-tokens-sum", type=int, help="Max number of tokens for generating the summary")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=34000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)

    return parser.parse_args()

def get_short_model_name(model_name):
    model_mapping = {
        "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
        "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
        "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",
        "Qwen/Qwen3-8B": "qwen3-8b",
        "Qwen/Qwen3-14B": "qwen3-14b",
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
        # for delimiter in ['\n\n', '\n---', '\nNote:', '\nAdditional']:
        #     if delimiter in summary:
        #         summary = summary.split(delimiter)[0]
        #         break
        
        return summary.strip("**\n\n")
    return text.strip()

def main():
    args = parse_args()
    load_dotenv(".env")
    hf_token = os.environ.get("HF_TOKEN")

    # Initialize the model
    if "Qwen" in args.model:
        # Use YARN to extend context length
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }
        llm = LLM(
            model = args.model,
            max_model_len = args.max_model_len,
            gpu_memory_utilization = args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_chunked_prefill=True,
            max_num_batched_tokens=args.max_num_batched_tokens,  # Reduce if OOM, increase for better throughput
            swap_space=4,   # GB of CPU memory for overflow
            enforce_eager=False,  # Keep as False for better performance
            download_dir="/mnt/ceph_rbd/llms",
            rope_scaling=rope_scaling,
            trust_remote_code=True
        )

    else:
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

    step1_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens_qa,
    )

    step2_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens_sum,
    )

    dataset_map = {"qmsum": "qmsum", "summscreen": "summ_screen_fd", "gov_report": "gov_report"}
    dataset = load_dataset("tau/scrolls", dataset_map[args.dataset], trust_remote_code=True)[args.split]
    data = dataset.select(range(min(args.max_samples, len(dataset))))
    documents = [item['input'] for item in data]

    question_prompt = """1. What are the important entities in this document?
    2. What are the important dates in this document?
    3. What events are happening in this document?
    4. What is the result of these events?
    Provide short answers containing only the most relevant entities to the questions above in complete sentences: """
    gen_prompt = {
        "gov_report": "Integrate the above information and write a one-page summary of the report. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.",
        "qmsum": "Integrate the above information and generate a summary in 4 sentences. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary."
    }
    # gen_prompt = "Integrate the above information and summarize the article. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary."

    print(f"Processing {len(documents)} documents with Summary CoT...")

    # Step 1: Element extraction
    print("Step 1: Extracting elements...")
    step1_prompts = [f"Article:\n{doc}\n\nQuestions:\n{question_prompt}" for doc in documents]
    step1_messages = [[{"role": "user", "content": prompt}] for prompt in step1_prompts]

    if "Qwen" in args.model:
        step1_outputs = llm.chat(messages=step1_messages,
                           sampling_params=step1_sampling_params,
                           use_tqdm=True,
                           chat_template_kwargs={"enable_thinking": False})
        
    else:
        step1_outputs = llm.chat(messages=step1_messages,
                           sampling_params=step1_sampling_params,
                           use_tqdm=True)
    
    element_responses = [output.outputs[0].text for output in step1_outputs]

    # Step 2: Final summary generation
    print("Step 2: Generating summaries...")
    step2_prompts = [
        f"Article:\n{doc}\n\nQuestions:\n{question_prompt}\n{ele_response}\n{gen_prompt[args.dataset]}"
        for doc, ele_response in zip(documents, element_responses)
    ]

    step2_messages = [[{"role": "user", "content": prompt}] for prompt in step2_prompts]
    
    if "Qwen" in args.model:
        step2_outputs = llm.chat(messages=step2_messages,
                           sampling_params=step2_sampling_params,
                           use_tqdm=True,
                           chat_template_kwargs={"enable_thinking": False})
        
    else:
        step2_outputs = llm.chat(messages=step2_messages,
                           sampling_params=step2_sampling_params,
                           use_tqdm=True)

    final_summaries = [extract_summary(output.outputs[0].text) for output in step2_outputs]
    
    # Save results
    results = []
    for i, (item, element_response, final_summary) in enumerate(zip(data, element_responses, final_summaries)):
        results.append({
            "id": item.get('id', i),
            "input": item['input'],
            "reference": item['output'],
            "element_extraction": element_response,
            "generated_summary": final_summary,
        })

    # Save to file
    os.makedirs(args.save_path, exist_ok=True)
    short_model_name = get_short_model_name(args.model)
    filename = os.path.join(args.save_path, f"{short_model_name}_{args.dataset}_{args.split}_{len(results)}_sum_cot.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated {len(results)} summaries → {filename}")

if __name__ == "__main__":
    main()