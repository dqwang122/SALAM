import os
import sys
import numpy as np
import pandas as pd
import argparse

import re
import json

import torch


from tqdm import tqdm

from utils.constants import sample_config, MODELNAME, greedy_config
from utils.tools import train_test_split, semantic_retrieval, simple_check_result
from llm import LLM

STOPSTRING = "\n"
REMOVESTR = "\n"

def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("--task_name", type=str, default="bbq_sm")
    args.add_argument("--data_root", type=str, default="data")
    args.add_argument("--root_dir", type=str, default="results")
    args.add_argument("--save_root", type=str, default=".")
    args.add_argument("--seed", type=int, default=42)

    # model options
    args.add_argument("--model", type=str, default="flan_t5", choices=['gpt2_neox', 'flan_t5', 'llama'])
    args.add_argument("--mode", type=str, default="zero_shot", choices=['zero_shot','few_shot'])
    args.add_argument("--device", type=str, default="cuda")

    # generate options
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--config", type=str, default="greedy", choices=['greedy', 'sample'])
    args.add_argument("--save", action='store_true', default=False)

    # retrieval options
    args.add_argument("--source", type=str, default='correct', choices=['mistake', 'correct', 'feedback', 'none', 'skip'])
    args.add_argument("--max_src_size", type=int, default=-1)
    args.add_argument("--fbfile", type=str, default='mistake_feedback.json')
    args.add_argument("--mkfile", type=str, default='mistake_collections.json')
    args.add_argument("--corrfile", type=str, default='correct_collections.json')
    args.add_argument("--topk", type=int, default=3)
    args.add_argument("--thre", type=float, default=0.9)


    args = args.parse_args()
    return args

def build_dataset(prompt, target):
    dataset = []
    for x, y in zip(prompt, target):
        ins = {'input': x, 'target': y}
        dataset.append(ins)
    return dataset

def response_from_student(model, test_query, target, test_target_opt, batch_size=1, stop_string=STOPSTRING, remove_str=REMOVESTR):
    dataset = build_dataset(test_query, target)
    response = model.generate(dataset, batch_size, stop_string=stop_string, remove_str=remove_str)
    score, detail = simple_check_result(response, test_target_opt)
    for i, (r, s) in enumerate(zip(response, detail)):
        r['index'] = i
        r['response/score'] = s
    return response, score



# python train.py --source correct --batch_size 1 --topk 1 --thre 0
# python train.py --task_name bbh --source skip --save --save_root results0520_bbh
# python train.py --task_name bbh --source correct --save_root results0520_bbh


if __name__ == "__main__":

    args = get_options()

    root_dir = args.root_dir
    data_root = args.data_root
    task_name = args.task_name
    save_root = args.save_root

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    mode = args.mode
    model_abbr = args.model
    device = args.device
    batch_size = args.batch_size
    config = greedy_config if args.config == 'greedy' else sample_config

    model_name = MODELNAME[model_abbr]
    print(model_abbr, model_name)

    status = train_test_split(root_dir, data_root, task_name, model_abbr, mode)

    
    if args.source == 'skip':
        # build a correct and mistake collection
        for k, v in status.items():
            train_data = v['train_set']
            train_score = v['train_score']
            v['train_mistake'] = []
            v['train_correct'] = []
            for i, (x, s) in enumerate(zip(train_data, train_score)):
                x['task'] = k
                x['task_idx'] = i
                if s == 0:
                    v['train_mistake'].append(x)
                else:
                    v['train_correct'].append(x)
                    
        mistake_collections = [x for k, v in status.items() for x in v['train_mistake']]
        correct_collections = [x for k, v in status.items() for x in v['train_correct']]

        if args.save:
            for i, x in enumerate(correct_collections):
                x['index'] = i
            for i, x in enumerate(mistake_collections):
                x['index'] = i
            json.dump(correct_collections, open(f"{save_root}/{model_abbr}_correct_collections.json", 'w'), indent=2)
            json.dump(mistake_collections, open(f"{save_root}/{model_abbr}_mistake_collections.json", 'w'), indent=2)
            sys.exit()
    else:
        mistake_collections = json.load(open(args.mkfile, 'r'))
        correct_collections = json.load(open(args.corrfile, 'r'))
        print("Correction collections:{} ({})".format(len(correct_collections), args.corrfile))
        print("Mistake collections:{} ({})".format(len(mistake_collections), args.mkfile))

    topk = args.topk
    thre = args.thre

    print('Building Model: ', model_name)
    llm = LLM(model_name=model_name, config=config, device=device)
    if args.source == 'none':
        # orignal test
        print('Original test')
        for name, v in status.items():
            test_data = v['test_set']
            test_target_opt = v['test_target_option']
            test_query = [x['response/prompt_str'] for x in test_data]
            target = [x['response/target_str'] for x in test_data]
            response, score = response_from_student(llm, test_query, target, test_target_opt, batch_size, stop_string=STOPSTRING, remove_str=REMOVESTR)
            ori_score = round(sum(v['test_score']) * 100 / len(v['test_score']), 2)
            print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )
            if args.save:
                json.dump(response, open(f"{save_root}/{name}_{model_abbr}_test.json", 'w'), indent=2)

    elif args.source == 'correct':
        # retrieve from correct collections
        collections = correct_collections
        print('Correct collections: ', len(collections))
        if args.max_src_size > 0:
            collections = collections[:args.max_src_size]   # 894
            print('Take Correct collections: ', len(collections))
        print('Topk: ', topk, 'Thre: ', thre)

        for name, v in status.items():
            test_data = v['test_set']
            test_target_opt = v['test_target_option']
            test_query = [x['response/prompt_str'] for x in test_data]
            corpus = [x['response/prompt_str'] for x in collections]
            ret = semantic_retrieval(test_query, corpus, top_k=topk)
            new_prompt = []
            target = [x['response/target_str'] for x in test_data]
            for q, r in zip(test_query, ret):
                icl = [collections[x['corpus_id']] for x in r if x['score'] > thre]
                ins = ["{}.\nThe answer is {}".format(x['response/prompt_str'], x['response/target_str']) for x in icl]
                q = "\n\n".join(ins) + "\n\n" + q + '\n' + 'The answer is '
                # print(q)
                # print('-'*100)
                new_prompt.append(q)

            response, score = response_from_student(llm, new_prompt, target, test_target_opt, batch_size, stop_string=STOPSTRING, remove_str=REMOVESTR)
            ori_score = round(sum(v['test_score']) * 100 / len(v['test_score']), 2)
            print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )

            if args.save:
                json.dump(response, open(f"{save_root}/{name}_{model_abbr}_correct_update_test.json", 'w'), indent=2)


    elif args.source == 'mistake':
        collections = mistake_collections
        print('Mistake collections: ', len(collections))
        if args.max_src_size > 0:
            collections = collections[:args.max_src_size]
            print('Take Mistake collections: ', len(collections))
        print('Topk: ', topk, 'Thre: ', thre)

        for name, v in status.items():
            test_data = v['test_set']
            test_target_opt = v['test_target_option']
            test_query = [x['response/prompt_str'] for x in test_data]
            corpus = [x['response/prompt_str'] for x in collections]
            ret = semantic_retrieval(test_query, corpus, top_k=topk)
            new_prompt = []
            target = [x['response/target_str'] for x in test_data]
            for q, r in zip(test_query, ret):
                icl = [collections[x['corpus_id']] for x in r if x['score'] > thre]
                ins = ["{}.\nPrevious wrong answer is {}. The correct answer is {}.".format(x['response/prompt_str'], x['response/output'],  x['response/target_str']) for x in icl]
                q = "\n\n".join(ins) + "\n\n" + q + '\n' + 'The correct answer is '
                # print(q)
                # print('-'*100)
                new_prompt.append(q)
                
            response, score = response_from_student(llm, new_prompt, target, test_target_opt, batch_size, stop_string=STOPSTRING, remove_str=REMOVESTR)
            ori_score = round(sum(v['test_score']) * 100 / len(v['test_score']), 2)
            print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )

            if args.save:
                json.dump(response, open(f"{save_root}/{name}_{model_abbr}_update_test_score.json", 'w'), indent=2)

    elif args.source == 'feedback':
        feedback = [json.loads(x) for x in open(args.fbfile, 'r')]
        print('Feedback collections: ', len(feedback))
        if args.max_src_size > 0:
            feedback = feedback[:args.max_src_size]
        n = len(feedback)
        print('Take Feedback collections: ', len(feedback))
        print('Topk: ', topk, 'Thre: ', thre)

        pattern = r"choose the '.*' option."
        for x in feedback:
            if re.findall(pattern, x['guideline']):
                print(re.findall(pattern, x['guideline']))
            x['guideline'] = re.sub(pattern, "choose the option which doesn't make a decision.", x['guideline'])

        collections = mistake_collections[:n]
        for name, v in status.items():
            # if name != 'Age_zero_shot':
            #     continue
            test_data = v['test_set']
            test_target_opt = v['test_target_option']
            test_query = [x['response/prompt_str'] for x in test_data]
            corpus = [x['response/prompt_str'] for x in collections]
            print('Corpus', len(corpus), 'Collections', len(collections))
            ret = semantic_retrieval(test_query, corpus, top_k=topk)
            new_prompt = []
            target = [x['response/target_str'] for x in test_data]
            for q, r in zip(test_query, ret):
                corpus_id = [x['corpus_id'] for x in r if x['score'] > thre]
                icl = [collections[x] for x in corpus_id]
                ins = ["{}.\nPrevious wrong answer is {}. The correct answer is {}.".format(x['response/prompt_str'], x['response/output'],  x['response/target_str']) for x in icl]
                q = "\n\n".join(ins) + "\n\n" + q + '\n' + 'The correct answer is '
                q = '\n'.join([feedback[x]['guideline'] for x in corpus_id]) + '\n\n'+ q
                # print(q)
                # print('-'*100)
                new_prompt.append(q)

            response, score = response_from_student(llm, new_prompt, target, test_target_opt, batch_size, stop_string=STOPSTRING, remove_str=REMOVESTR)
            ori_score = round(sum(v['test_score']) * 100 / len(v['test_score']), 2)
            print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )

            if args.save:
                json.dump(response, open(f"{save_root}/{name}_{model_abbr}_f_update_test.json", 'w'), indent=2)

    else:
        raise NotImplementedError
