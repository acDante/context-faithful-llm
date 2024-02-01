import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import datasets
import evaluate
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from models import context_aware_wrapper
from metrics import mean_score


input_key = {
    "xsum": "document",
    "cnn_dm": "article"
}

output_key = {
    "xsum": "summary",
    "cnn_dm": "highlights"
}

def get_question_prompt(doc, schema, summary=""):
    if schema == "base":
        prompt = f'Summary:{summary}'
    elif schema == "opin":
        prompt = f'Summarise the above article in Bob\'s opinion. Summary:{summary}'
    elif schema == "attr":
        prompt = f'Summarise the above article. Summary:{summary}'
    elif schema == 'instr':
        prompt = f'Summary:{summary}'
    elif schema == 'instr+opin':
        prompt = f'Summarise the above article in Bob\'s opinion. Summary:{summary}'

    return prompt

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

    return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dm", type=str, choices=['cnn_dm', 'xsum'])
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num_samples", default=2500, type=int, help="Number of test samples to evaluate on")
    parser.add_argument("--use_cad", action="store_true", help="Use context-aware decoding")
    parser.add_argument("--alpha", default=1.0, type=float, help="Parameter for context-aware decoding")
    parser.add_argument("--schema", default="base", type=str, choices=['base', 'opin', 'instr', 'attr', 'instr+opin'],
                        help="which context faithful prompting format to use")
    parser.add_argument("--log_path", default="results/", type=str)
    parser.add_argument("--exp_name", type=str, help="Experiment name")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    access_token = "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"
    download_path = "/home/hpcdu1/experiments/huggingface-hub"

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 token=access_token)
                                                #  cache_dir=download_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=access_token)
                                            #   cache_dir=download_path)
    
    if args.use_cad:
        model = context_aware_wrapper(model, alpha=args.alpha, k=1)

    # Load FactKB model
    factkb_tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
    factkb_model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2, device_map="auto")

    if args.dataset == "cnn_dm":
        test_data = datasets.load_dataset('cnn_dailymail', '3.0.0', split='test')
    elif args.dataset == 'xsum':
        test_data = datasets.load_dataset("xsum", split="test")

    test_data = test_data.select(range(min(args.num_samples, len(test_data))))  # Use the first few samples for faster iteration

    factkb_scores = []
    pred_samples = []
    predictions = []
    references = []

    # Q: test data contains about 10k samples, how to speed up inference?
    for idx, sample in tqdm(enumerate(test_data)):
        doc = sample[input_key[args.dataset]]
        reference = sample[output_key[args.dataset]]
        # Q: what prompt format to use?
        prompt = get_prompt(doc, args.schema)
        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = tokenizer.apply_chat_template(messages, truncation=True, return_tensors="pt")
        input_ids = input_ids.to("cuda")

        # Issue: model tends to generate a very long summary. when to stop?
        # Q: How to set proper hyperparameters for summarisation?
        # TODO: try stopping criteria
        if args.use_cad:
            ques_messages = [
                {"role": "user", "content": get_question_prompt(doc, args.schema)}
            ]
            question_ids = tokenizer.apply_chat_template(ques_messages, truncation=True, return_tensors="pt")
            question_ids = question_ids.to("cuda")
            model.update(question_ids.shape[-1])

        outputs = model.generate(input_ids,
                                 do_sample=False,
                                 max_new_tokens=256,
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

        # Evaluate FactKB score
        factkb_input = [[output, doc]]
        factkb_tokens = factkb_tokenizer(factkb_input, return_tensors="pt", 
                                         padding="max_length", truncation=True).to(factkb_model.device)
        factkb_logits = factkb_model(**factkb_tokens).logits
        factkb_res = torch.softmax(factkb_logits, dim=1)
        factkb_scores.append(float(factkb_res[0][1]))

    # Evaluate the performance:  https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/blob/main/src/backend/tasks/xsum/task.py#L55

    breakpoint()
    # Compute the ROUGE scores -> Q: which ROUGE implementation to use?
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=predictions, 
                                 references=references,
                                 use_aggregator=False)

    # Compute BERT score
    # TODO: Debug bert score: when using CAD, the process is killed here for unknown reason
    bert_score = evaluate.load('bertscore')
    bert_score_res = bert_score.compute(predictions=predictions, 
                                        references=references, 
                                        model_type="microsoft/deberta-xlarge-mnli", lang="en")

    # Aggregate the evaluation metrics
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metrics = {
        'rouge1': mean_score(rouge_scores['rouge1']),
        'rouge2': mean_score(rouge_scores['rouge2']),
        'rougeLsum': mean_score(rouge_scores['rougeLsum']),
        'bertscore_p': mean_score(bert_score_res['precision']),
        'bertscore_r': mean_score(bert_score_res['recall']),
        'bertscore_f1': mean_score(bert_score_res['f1']),
        "factKB": mean_score(factkb_scores)
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
                fout.write("\n")


if __name__ == '__main__':
    main()
