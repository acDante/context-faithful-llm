import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def get_prompt(doc, schema, summary=""):
    """Context faithful prompting for summarisation"""
    prompt = ""
    if schema == "base":
        prompt = f'Document: {doc}\nSummary:{summary}'
    elif schema == "opin":
        prompt = f'Bob said "{doc}"\nSummarise the above article in Bob\'s opinion. Summary:{summary}'
    elif schema == "attr":
        prompt = f'Document: {doc}\nSummarise the above article. Summary:{summary}'
    elif schema == 'instr':
        prompt = f'Instruction: read the given article and provide a brief summary.\n\nDocument: {doc}\nSummary:{summary}'
    elif schema == 'instr+opin':
        prompt = f'Instruction: read the given article and provide a brief summary.\n\nBob said "{doc}"\nSummarise the above article in Bob\'s opinion. Summary:{summary}'
        pass  # TODO

    return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dm", type=str, choices=['cnn_dm', 'xsum'])
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--schema", default="base", type=str, choices=['base', 'opin', 'instr', 'attr', 'instr+opin'])
    parser.add_argument("--log_path", default="results/", type=str)
    parser.add_argument("--exp_name", type=str, default="Experiment name")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    access_token = "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"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=access_token)
    
    test_data = datasets.load_dataset('cnn_dailymail', '3.0.0', split='test')

    pred_samples = []
    predictions = []
    references = []
    for idx, sample in tqdm(enumerate(test_data)):
        doc = sample['article']
        reference = sample['highlights']
        # Q: what prompt format to use?
        prompt = get_prompt(doc, args.schema)
        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_ids = input_ids.to("cuda")

        # Issue: model tends to generate a very long summary. when to stop?
        # Q: How to set proper hyperparameters for summarisation?
        outputs = model.generate(input_ids,
                                 do_sample=False,
                                 max_new_tokens=1000,
                                 temperature=0.0)
        raw_output = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        output = raw_output.split("\n")[0]

        predictions.append(output)
        references.append(reference)
        
        # Save predictions to file
        pred_sample = dict()
        pred_sample['article'] = doc
        pred_sample['highlights'] = reference
        pred_sample['prediction'] = output
        pred_samples.append(pred_sample)

        if idx == 25:  # for debugging
            break
    
    # Evaluate the performance:  https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/blob/main/src/backend/tasks/xsum/task.py#L55

    # Compute the ROUGE scores -> Q: which ROUGE implementation to use?
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, 
                            references=references,
                            use_aggregator=False)

    # Compute BERT score
    bert_score = evaluate.load('bertscore')
    bert_score_res = bert_score.compute(predictions=predictions, 
                                        references=references, 
                                        model_type="microsoft/deberta-xlarge-mnli", lang="en")

    # Compute FactKB score 
    # TODO: Add FactKB score and/or other hallucination metrics?

    # Aggregate the evaluation metrics
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metrics = {
        'rouge1': sum(results['rouge1']) / len(results['rouge1']),
        'rouge2': sum(results['rouge2']) / len(results['rouge2']),
        'rougeLsum': sum(results['rougeLsum']) / len(results['rougeLsum']),
        'bertscore_f1': sum(bert_score_res['f1']) / len(bert_score_res['f1'])
    }
    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}_preds.json", 'w') as fh:
            json.dump(pred_samples, fh, indent=4)

    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}.log", "w") as fout:
            for metric_name, value in metrics.items():
                fout.write(f"{metric_name}: {value:.4f}")
                fout.write("\t")

if __name__ == '__main__':
    main()
