import click
import json
import torch
import argparse
import os
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import transformers

from model import StudyAssistant
from utils.constants import STOPSTRING, REMOVESTR, TEMPLATE, TEMPLATE_TEST, SAMPLE_CONFIG



MAX_SOURCE_LENGTH = 2048
MAX_TARGET_LENGTH = 512
print("Max source length: ", MAX_SOURCE_LENGTH)
print("MAX target length: ", MAX_TARGET_LENGTH)


def get_input():
    mode = input("Choose your mode (feedback / agent / quit): ")
    while mode not in ["feedback", "agent", "quit"]:
        mode = input("Rechoose your mode (feedback / agent / quit): ")
    if mode == "feedback":
        print("In the feedback mode, the study assistant will have the query, the ground truth answer and the model response, and it will provide a feedback based on the response. ")
        query = input(f"Please input the query: ")
        wrong_answer = input(f"Please input the wrong response: ")
        correct_answer = input(f"Please input the correct answer")
        ins = {
            'query': query,
            'respose': wrong_answer,
            'target': correct_answer
        }
        return mode, ins
    elif mode == 'agent':
        print("In the agent mode, the study assistant only has the query, and it will predict a potential mistake and give the corresponding feedback.")
        query = input(f"Please input the query: ")
        ins = {
            'query': query,
        }
        return mode, ins
    else:
        return None, None

def check_input_format(mode, x):
    assert isinstance(x, dict)
    if mode == 'feedback':
        assert 'query' in x.keys() and isinstance(x['query'], str)
        assert 'respose' in x.keys() and isinstance(x['respose'], str)
        assert 'target' in x.keys() and isinstance(x['target'], str)
    elif mode == 'agent':
        assert 'query' in x.keys() and isinstance(x['query'], str)
    else:
        raise NotImplementedError

def make_prompt(mode, ins):
    check_input_format(mode, x=ins)
    if mode == 'feedback':
        return TEMPLATE.format(question=ins['query'], ans=ins['respose'], target=ins['target'])
    elif mode == 'agent':
        return TEMPLATE_TEST.format(question=ins['query'])
    else:
        raise NotImplementedError


def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("--mode", type=str, default='evaluate', choices=['test', 'interactive'])
    args.add_argument("--input_file", type=str, default="data/bbh/flan_t5/mistake_collections.json")
    args.add_argument("--save_dir", type=str, default="results/")
    args.add_argument("--seed", type=int, default=42)

    # model options
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--ckpt", type=str, default="checkpoint/checkpoint-217")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--temperature", type=float, default=0.8)


    args = args.parse_args()
    return args


# python src/inference.py --mode interactive --ckpt checkpoints/bbh_llama_0607/checkpoint-928/
# python src/inference.py --mode test --input_file data/bbh/flan_t5/mistake_collections.json --ckpt checkpoints/bbh_llama_0607/checkpoint-928/

if __name__ == "__main__":
    args = get_options()

    if args.mode == 'test':
        data = json.load(open(args.input_file, 'r'))
        print(f"Load {len(data)} from {args.input_file}")
        prompts = []
        for v in data:
            prompt = TEMPLATE.format(question=v['response/prompt_str'], ans=v['response/output'], target=v['response/target_str'])
            prompts.append(prompt)
    elif args.mode == 'interactive':
        mode, ins = get_input()
        if mode is None or ins is None:
            exit()
        prompts = make_prompt(mode, ins)
        print(prompts)
        prompts = [prompts]
    else:
        raise NotImplementedError
    
    # load model
    sa = StudyAssistant(model_name_or_path=args.ckpt, device="cuda")

    # inference
    print("Start inference for {} examples".format(len(prompts)))
    configs = SAMPLE_CONFIG
    responses = sa.inference(prompts, batch_size=args.batch_size, **configs)

    # save results
    save_file = f"{args.save_dir}/{args.ckpt.strip('/').split('/')[-1]}.json"
    print(f"Save to {save_file}")
    with open(save_file, "w") as f:
        for p, r in zip(prompts, responses):
            res = {
                'prompt': p,
                'response': r
            }
            f.write(json.dumps(res) + "\n")
