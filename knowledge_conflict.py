import json
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from engine import Engine, LLamaModel
from evaluation import get_score


def get_question_prompt(query, schema, answer=''):
    if schema == 'base':
        prompt = 'Q:{}\nA:{}'.format(query, answer)
    elif schema == 'opin':
        prompt = 'Q: {} in Bob\'s opinion?\nA:{}'.format(query[:-1], answer)
    elif schema == 'instr+opin':
        prompt = 'Q: {} in Bob\'s opinion?\nA:{}'.format(query[:-1], answer)
    elif schema == 'attr':
        prompt = 'Q:{} based on the given text?\nA:{}'.format(query[:-1], answer)
    elif schema == 'instr':
        prompt = 'Q:{}\nA:{}'.format(query, answer)

    return prompt

def qa_to_prompt(query, context, schema, demos=[], num_demos=16):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given text?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'no_context':
            prompt = f'Q:{query}\nA:{answer}'
        else:
            prompt = ""
        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    for demo in demos[-num_demos:]:
        answer = demo['answer'] if isinstance(demo['answer'], str) else demo['answer'][0]
        demo_prompt = get_prompt(demo['question'], demo['context'], schema=schema, answer=answer)
        prompt = prompt + demo_prompt + '\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    return prompt

def eval(pred_answers, orig_answers, gold_answers):
    em, ps = get_score(pred_answers, gold_answers)
    _, po = get_score(pred_answers, orig_answers)
    mr = po / (ps + po + 1e-10) * 100
    print('ps {}, po {}, mr {}, em {}.'.format(ps, po, mr, em))

    metrics = {
        "ps": ps,
        "po": po,
        "mr": mr,
        "em": em
    }

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", default="./datasets/nq/orig_dev_filtered.json", type=str)
    parser.add_argument("--counter_path", default="./datasets/nq/conflict_dev_filtered.json", type=str)
    parser.add_argument("--engine", default="text-davinci-003", type=str)
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--demo_mode", default="none", help="Choose from the following demonstrations: none, original, counter.")
    parser.add_argument("--num_demos", default=16, type=int)
    parser.add_argument("--use_cad", action="store_true", help="Use context-aware decoding")
    parser.add_argument("--alpha", default=1.0, type=float, help="Parameter for context-aware decoding")
    parser.add_argument("--log_path", default='results/', type=str)
    parser.add_argument("--exp_name", type=str, default="Experiment name")

    args = parser.parse_args()
    with open(args.orig_path, 'r') as fh:
        orig_examples = json.load(fh)
    with open(args.counter_path, 'r') as fh:
        counter_examples = json.load(fh)
    print('Loaded {} instances.'.format(len(counter_examples)))
    
    # engine = Engine(args.engine)
    # Use open-source models
    if args.use_cad:
        engine = LLamaModel(args.model_name, use_cad=True, alpha=args.alpha)
    else:
        engine = LLamaModel(args.model_name)

    step = 0
    gold_answers, pred_answers, orig_answers = [], [], []
    predictions = []   # save predictions
    for oe, ce in tqdm(zip(orig_examples, counter_examples), total=len(orig_examples)):
        if step % 100 == 0:
            eval(pred_answers, orig_answers, gold_answers)
        step += 1
        query, context, answer = ce['question'], ce['context'],ce['answer']
        orig_answer = oe['answer']
        # Note: use original answer to filter the test data, comment this for normal experiments
        # answer = orig_answer
        if orig_answer is None:
            continue
        if args.demo_mode == 'none':
            demos = []
        elif args.demo_mode == 'counter':
            demos = ce['ic_examples']
        elif args.demo_mode == 'original':
            demos = ce['ico_examples']

        # prompt = qa_to_prompt(query, context, schema=args.schema, demos=demos, num_demos=args.num_demos)
        prompt = ""
        #  CAD Decoding
        if args.use_cad:
            for num_demos in range(args.num_demos, 1, -1):  # Use fewer demos if prompt is too long
                prompt = qa_to_prompt(query, context, schema=args.schema, demos=demos, num_demos=num_demos)
                if not engine.check_prompt_length(prompt):
                    break
            if engine.check_prompt_length(prompt):  # Truncate long context that exceeds max input length
                continue
            
            ques_prompt = get_question_prompt(query, schema=args.schema)
            pred = engine.complete(prompt, ques_prompt)

        # Greedy decoding
        else:  
            for num_demos in range(args.num_demos, 1, -1):  # Use fewer demos if prompt is too long
                prompt = qa_to_prompt(query, context, schema=args.schema, demos=demos, num_demos=num_demos)
                if not engine.check_prompt_length(prompt):
                    break
            if engine.check_prompt_length(prompt):  # Truncate long context that exceeds max input length
                continue

            pred = engine.complete(prompt)
        if pred is None:
            pred = ""
            # continue
        pred_answers.append(pred)
        gold_answers.append(answer)
        orig_answers.append(orig_answer)
        # Logs
        ce['prediction'] = pred
        ce['orig_answer'] = orig_answer
        ce['schema'] = args.schema
        ce['demo_mode'] = args.demo_mode

        # Save predictions to file
        pred_sample = dict()
        pred_sample['question'] = query
        pred_sample['context'] = context
        pred_sample['answer'] = answer
        pred_sample['prediction'] = pred

        predictions.append(pred_sample)
    
    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}_preds.json", 'w') as fh:
            json.dump(predictions, fh, indent=4)
            # json.dump(counter_examples, fh, indent=4)
    
    metrics = eval(pred_answers, orig_answers, gold_answers)
    if args.log_path:
        log_path = Path(args.log_path)
        with open(log_path / f"{args.exp_name}.log", "w") as fout:
            for metric_name, value in metrics.items():
                fout.write(f"{metric_name}: {value:.4f}")
                fout.write("\t")

if __name__ == '__main__':
    main()