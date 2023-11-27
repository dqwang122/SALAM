import os
import sys
import numpy as np
import pandas as pd
import argparse

import re
import json
from tqdm import tqdm

import torch

from utils.tools import train_test_split
from utils.constants import GREEDY_CONFIG, SAMPLE_CONFIG

from model import SALAM

CHOICES=['mistake', 'correct', 'feedback', 'feedback_cot', 'feedback_only', 'agent', 'none',]

def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("--mode", type=str, default="preprocess", choices=['exam', 'gather'])
    args.add_argument("--task_name", type=str, default="bbq_sm")
    args.add_argument("--data_root", type=str, default="../data")
    args.add_argument("--save_root", type=str, default="../results")
    args.add_argument("--seed", type=int, default=42)

    # model options
    args.add_argument("--llm_model", type=str, default="flan_t5", choices=['gpt2_neox', 'flan_t5', 'llama'])
    args.add_argument("--sa_model", type=str, default="yahma/llama-7b-hf")
    args.add_argument("--device", type=str, default="cuda")

    # generate options
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--config", type=str, default="greedy", choices=['greedy', 'sample'])
    args.add_argument("--save", action='store_true', default=False)

    # retrieval options
    args.add_argument("--source", type=str, default='correct', choices=CHOICES)
    args.add_argument("--max_mem_size", type=int, default=-1)
    args.add_argument("--max_iteration", type=int, default=1)
    args.add_argument("--fbfile", type=str, default='mistake_feedback.json')
    args.add_argument("--topk", type=int, default=3)
    args.add_argument("--thre", type=float, default=0.9)


    args = args.parse_args()
    return args


# python run.py --mode preprocess --task_name bbq_sm --llm_model flan_t5 --save --save_root ../data 
# python run.py --mode exam --task_name bbh --source correct --llm_model flan_t5 --save_root ../results/1126
# python run.py --mode exam --task_name bbh --source correct --llm_model flan_t5 --sa_model checkpoints/bbh/ --save_root ../results/1126

if __name__ == "__main__":

    args = get_options()

    data_root = args.data_root
    task_name = args.task_name
    save_root = args.save_root

    if not os.path.exists(f'{save_root}/{task_name}'):
        os.makedirs(f'{save_root}/{task_name}')
    
    memory_dir = f"{data_root}/{args.task_name}/{args.llm_model}"
    status  = json.load(open(f"{memory_dir}/status.json", 'r'))

    salam = SALAM(basemodel=args.llm_model, study_assistant=args.sa_model, 
                  memory_dir=memory_dir, save_root=args.save_root, 
                  device=['cuda:0', 'cuda:1'], 
                  llm_config=GREEDY_CONFIG if args.config == 'greedy' else SAMPLE_CONFIG)
    

    if args.mode == 'exam':
        if args.save:
            if not os.path.exists(f'{save_root}/{task_name}'):
                os.makedirs(f'{save_root}/{task_name}')
                
        salam.examination(data=status, source=args.source, 
                          max_mem_size=args.max_mem_size, feedback_file=args.fbfile, 
                          batch_size=args.batch_size, topk=args.topk, 
                          thre=args.thre, save=args.save)

    elif args.mode == 'gather':
        if not os.path.exists(f'{save_root}/{task_name}'):
            os.makedirs(f'{save_root}/{task_name}')

        salam.gather(data=status, batch_size=args.batch_size, 
                     topk=args.topk, thre=args.thre, 
                     max_iteration=args.max_iteration, 
                     max_mem_size=args.max_mem_size, save=True)

    else:
        raise NotImplementedError

    
    

