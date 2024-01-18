import json
from pathlib import Path
from engine import Engine, LLamaModel, Llama2Model
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import tiktoken
import argparse
from sklearn.metrics import brier_score_loss
from transformers import LlamaTokenizer

def qa_to_prompt(query, context, choices, schema, answer=''):
    context = context.replace('“', '"').replace('”', '"').replace('’', "'")
    if schema == 'base':
        prompt = '{}\n\nQ: {}\nChoices: {}\nA: {}'.format(context, query, choices, answer)
    elif schema == 'opin':
        context = context.replace('"', "")
        prompt = 'Bob said, "{}"\n\nQ: {} in Bob\'s opinion?\nChoices: {}\nA: {}'.format(context, query[:-1], choices, answer)
    elif schema == 'attr':
        prompt = '{}\n\nQ:{} based on the given text?\nChoices: {}\nA: {}'.format(context, query[:-1], choices, answer)
    elif schema == 'instr':
        prompt = '{}\n\nQ: {}\nChoices: {}\nA: {}'.format(context, query, choices, answer)
    elif schema == 'instr+opin':
        context = context.replace('"', "")
        prompt = 'Bob said, "{}"\n\nQ: {} in Bob\'s opinion?\nChoices: {}\nA: {}'.format(context, query[:-1], choices, answer)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./datasets/realtime_qa/realtime_qa_data.json", type=str)
    parser.add_argument("--demo_path", default="./datasets/realtime_qa/realtime_qa_demo_data.json", type=str)
    parser.add_argument("--engine", default="text-davinci-003", type=str)
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--demo_mode", default="none", help="Choose from the following demonstrations: none, original.")
    parser.add_argument('--ans_mode', default='mean', type=str)
    parser.add_argument("--log_path", default='results/', type=str)
    parser.add_argument("--exp_name", type=str, default="Experiment name")

    args = parser.parse_args()
    with open(args.data_path, 'r') as fh:
        test_data = json.load(fh)
    with open(args.demo_path, 'r') as fh:
        demo_data = json.load(fh)
    # engine = Engine(args.engine)
        
    # Use Llama-2 model instead of GPT
    engine = LLamaModel(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    abs_golds, abs_probs, preds, golds = [], [], [], []
    predictions = []  # save predictions
    for d in tqdm(test_data):
        context, question, choices, answer = d['context'], d['question'], d['choices'], d['answer']
        probs = []
        for choice in choices.split(';'):
            choice = choice.strip()
            assert len(choice) > 0
            prompt = ''
            if args.schema in ('instr', 'instr+opin'):
                prompt = 'Instruction: answer a question based on the provided input-output pairs.\n\n'
            if args.demo_mode == 'original':
                for demo in demo_data:
                    prompt += (qa_to_prompt(demo['question'], demo['context'], demo['choices'], args.schema, answer=demo['answer']) + '\n\n')
            
            choice = choice.strip() + '.'
            prompt += qa_to_prompt(question, context, choices, args.schema)
            prompt = prompt + choice
            if engine.check_prompt_length(prompt):
                continue
            num_tokens = len(tokenizer(choice)['input_ids']) - 1
            prob = engine.get_prob(prompt, num_tokens)
            if prob is not None:
                if args.ans_mode == "mean":
                    prob = prob / num_tokens
                elif args.ans_mode == 'cond':
                    c_prob = engine.get_prob(' ' + choice)
                    prob = prob - c_prob
                probs.append(prob)

        if len(probs) != len(choices.split(';')):
            continue
        choice_probs = softmax(np.array(probs))
        choices = [s.strip() for s in choices.split(';')]
        pred = choices[probs.index(max(probs))]
        d['pred'] = pred
        d['probs'] = choice_probs.tolist()
        abs_gold = 1 if answer == 'I don\'t know' else 0
        abs_golds.append(abs_gold)
        abs_probs.append(choice_probs.tolist()[-1])
        preds.append(pred)
        golds.append(answer)

        # Save predictions to file
        pred_sample = dict()
        pred_sample['context'] = context
        pred_sample['question'] = question
        pred_sample['choices'] = choices
        pred_sample['answer'] = answer
        pred_sample['pred'] = pred
        pred_sample['probs'] = choice_probs.tolist()

        predictions.append(pred_sample)

    # Evaluation
    has_ans_correct, no_ans_correct, has_ans_wrong, no_ans_wrong = 0, 0, 0, 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            if gold != 'I don\'t know':
                has_ans_correct += 1
            else:
                no_ans_correct += 1
        else:
            if gold != 'I don\'t know':
                has_ans_wrong += 1
            else:
                no_ans_wrong += 1
        
    hasans_acc = has_ans_correct / (has_ans_correct + has_ans_wrong)
    noans_acc = no_ans_correct / (no_ans_correct + no_ans_wrong)
    acc = (has_ans_correct + no_ans_correct) / (has_ans_correct + has_ans_wrong + no_ans_correct + no_ans_wrong)
    brier = brier_score_loss(np.array(abs_golds), np.array(abs_probs))
    print("HasAns Acc {}, NoAns Acc {}, Acc {}, Brier {}.".format(hasans_acc, noans_acc, acc, brier))
    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}_preds.json", 'w') as fh:
            json.dump(test_data, fh, indent=4)
    
    # Save predictions to file
    # if args.log_path:
    #     log_path = Path(args.log_path)
    #     with open(log_path / f"{args.exp_name}_preds.json", "w") as fh:
    #         json.dump(predictions, fh, indent=4)

    # Save evaluation metrics in log file
    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}.log", "w") as fout:
            fout.write(f"HasAns Acc {hasans_acc}, NoAns Acc {noans_acc}, Acc {acc}, Brier {brier}.")
            fout.write("\t")


if __name__ == '__main__':
    main()