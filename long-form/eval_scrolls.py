import argparse
import json
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summaries on long-form datasets using vLLM")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", 
                        help="Model name")
    parser.add_argument("--dataset", choices=["qmsum", "summscreen", "gov_report"], default="qmsum",
                        help="Dataset used for evaluation")
    parser.add_argument("--save-path", default="results/summary", help="Path to save the predictions")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=34000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attr_data_path", type=str, default=None, help="Path to the processed data with annotated attributions")
    parser.add_argument("--attr_type", type=str, default="sent3", help="Which attribution method is used")
    parser.add_argument("--method", type=str, default="base", choices=['base', 'base+impt', 'base+impt_prefix', 'sum_cot'], help="which attribution-guided generation approach to use")

    return parser.parse_args()

def get_prompt(doc, important_sents, args):
    if args.method == "base": 
        prompt_template = {
            "qmsum": f"Read the following meeting transcript. Produce a summary in 4 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{doc}\nNow generate the summary in 4 sentences: ",
            "summscreen": f"Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{doc}\nNow generate the summary in 5 sentences: ",
            "gov_report": f"You are given a report by a government agency. Write a one-page summary of the report. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n\nReport:\n{doc}"
        }

    elif args.method == "base+impt":
        key_points = "\nYou should only focus on the following key points:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(important_sents)]) + "\n"
        prompt_template = {
            "qmsum": f"Read the following meeting transcript. Produce a summary in 4 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{doc}\n{key_points}\nNow generate the summary in 4 sentences: ",
            "summscreen": f"Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{doc}\n{key_points}\nNow generate the summary in 5 sentences: ",
            "gov_report": f"You are given a report by a government agency. Write a one-page summary of the report focusing on the main points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n\nReport:\n{doc}\n{key_points}"
        }

    elif args.method == "base+impt_prefix":
        key_points = "\nYou should only focus on the following key points:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(important_sents)]) + "\n"
        prompt_template = {
            "qmsum": f"Read the following meeting transcript. Produce a summary in 4 sentences focusing on key decisions, action items, and important discussion points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.{key_points}\n==========\n[MEETING TRANSCRIPT]\n==========\n{doc}\nNow generate the summmary in 4 sentences: ",
            "summscreen": f"Read the following TV episode transcript. Produce a summary in 5 sentences focusing on the main plot developments and key story events. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.{key_points}\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{doc}\nNow generate the summary in 5 sentences: ",
            "gov_report": f"You are given a report by a government agency. Write a one-page summary of the report focusing on the main points. You must give your answer in a structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.{key_points}\nReport:\n{doc}"
        }

    elif args.method == "sum_cot":
        prompt_questions = "1. What are the important entities in this document?\n2. What are the important dates in this document?\n3. What events are happening in this document?\n4. What is the result of these events?\n"
        prompt_template = {
            "qmsum": f"Read the following meeting transcript. Answer the quetions below and then produce a summary in 4 sentences by integrating the information in your answers. You must give your response in a structured format: \"Answers:\n1. answer1, 2. answer2, ...\nSummary: [your summary]\", where [your summary] is your generated summary.\n==========\n[MEETING TRANSCRIPT]\n==========\n{doc}\n{prompt_questions}",
            "summscreen": f"Read the following TV episode transcript. Answer the questions below and then produce a summary in 5 sentences by integrating the information in your answers. You must give your response in a structured format: \"Answers:\n1. answer1, 2. answer2, ...\nSummary: [your summary]\", where [your summary] is your generated summary.\n==========\n[TV EPISODE TRANSCRIPT]\n==========\n{doc}\n{prompt_questions}",
            "gov_report": f"You are given a report by a government agency. Answer the questions below and then write a one-page summary of the report by integrating the information in your answers. You must give your response in a structured format: \"Answers:\n1. answer1, 2. answer2, ...\n Summary: [your summary]\", where [your summary] is your generated summary. \n\nReport:\n{doc}\n\nQuestions:\n{prompt_questions}\nProvide short answers containing only the most relevant entities to the questions without using bullet points and then generate the summary after \"Summary\" prompt word: "
        }

    return prompt_template[args.dataset]
    # return prompt_template[args.dataset].format(doc)

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

def extract_answers_and_summary(text):
    """
    Extract answers and summary from LLM output. (for summary CoT)
    
    Args:
        text (str): The LLM output text containing answers and summary
        
    Returns:
        tuple: (list of answers to the predefined questions, summary string)
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
        
        # Parse the answers into a list
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

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Load the dataset
    if args.attr_data_path:
        with open(args.attr_data_path, "r") as fin:
            data = json.load(fin)
            data = data[:args.max_samples]
    else:
        dataset_map = {"qmsum": "qmsum", "summscreen": "summ_screen_fd", "gov_report": "gov_report"}
        # Load non query-based QMSum test data
        # if args.dataset == "qmsum":
        #     data_path = "/mnt/ceph_rbd/datasets/QMSum/processed_data/test.jsonl"
        #     dataset = []
        #     with open(data_path, 'r', encoding='utf-8') as f:
        #         for line in f:
        #             line = line.strip()
        #             if line:
        #                 dataset.append(json.loads(line))
        #     data = dataset[:args.max_samples]
        # else:
        dataset = load_dataset("tau/scrolls", dataset_map[args.dataset], trust_remote_code=True)[args.split]
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
        if args.attr_data_path:
            attributed_sents =  [sent['input_sequence'] for sent in item['attributed_sents'] if sent['score'] > 0]
        else:
            attributed_sents = None
        prompt = get_prompt(doc, attributed_sents, args)
        chat_messages.append([
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt}
        ])


    # Generate summaries
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
    
    print("Output: ", len(outputs))

    if args.method == "sum_cot":
        raw_outputs = []
        for output in outputs:
            raw_outputs.append(output.outputs[0].text)
        
        results = []
        for i, (item, prediction) in enumerate(zip(data, raw_outputs)):
            answers, summary = extract_answers_and_summary(prediction)
            result_item = dict(item)
            result_item['answers'] = answers
            result_item['generated_summary'] = summary
            result_item['raw_output'] = prediction
            results.append(result_item)

    else:
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
    if args.attr_data_path:
        filename = os.path.join(args.save_path, f"{short_model_name}_{args.dataset}_{args.split}_{len(results)}_attr-{args.attr_type}_{args.method}.json")
    else:
        filename = os.path.join(args.save_path, f"{short_model_name}_{args.dataset}_{args.split}_{len(results)}_{args.method}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Generated {len(results)} summaries → {filename}")

if __name__ == "__main__":
    main()
