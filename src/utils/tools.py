import re
import os
import json

from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

def check_result(results, dataset, save_path=None):
    if save_path is not None:
        fout = open(save_path, 'a+')
        fout.write(f'{results}\t|\t{dataset}\n')
    accu_cnt = 0
    score = []
    target_option_list = []
    for res,req in zip(results, dataset):
        output = res['response/output'].strip()
        target_option = res['response/target_str'].strip()
        if not (target_option.startswith('(') and target_option.endswith(')')):
            print(target_option)
        options = req['options']
        option_dict = {opt.split(' ')[0]: opt for opt in options}
        if target_option in option_dict:
            target_option = option_dict[target_option]

        output = output.lstrip('A: ')
        output = output.rstrip('.')
        output = output.lower()
        output = output.replace('\n', '').replace(u'）', ')').replace(u'（', '(')
        target_option = target_option.lower()
        matchobj = re.search(r'\([a-z]\)', output)
        if matchobj:
            output = matchobj.group(0)
        if output != "":
            if output in target_option or target_option in output:
                accu_cnt += 1
                score.append(1)
            else:
                score.append(0)
        else:
            score.append(0)
        # print(output + '\t|\t' + target_option + '\t|\t' + str(accu_cnt))
        target_option_list.append(target_option)
        if save_path is not None:
            fout.write(output + '\t|\t' + target_option + '\t|\t' + str(accu_cnt) + '\n')

    # print(results)
    return round(accu_cnt * 100 / len(results), 2), score, target_option_list

def simple_check_result(results, targets):
    accu_cnt = 0
    score = []
    for res,target_option in zip(results, targets):
        output = res['response/output'].strip()
        output = output.lstrip('A: ')
        output = output.rstrip('.')
        output = output.lower()
        output = output.replace('\n', '').replace(u'）', ')').replace(u'（', '(')
        target_option = target_option.lower()
        matchobj = re.search(r'\([a-z]\)', output)
        if matchobj:
            output = matchobj.group(0)
        if output != "":
            if output in target_option or target_option in output:
                accu_cnt += 1
                score.append(1)
            else:
                score.append(0)
        else:
            score.append(0)
    return round(accu_cnt * 100 / len(results), 2), score


def train_test_split(root_dir, data_root, task_name, model_name, mode):
    print(f'{root_dir} | {data_root} | {task_name} | {mode} | {model_name}')
    result_path = f"{root_dir}/{task_name}/{model_name}"
    data_path = f"{data_root}/{task_name}"
    file_names = [x for x in sorted(os.listdir(result_path)) if x.endswith(f'{mode}.json')]
    result_files = [(os.path.join(result_path, x), os.path.join(data_path, x)) for x in file_names]
    status = {}
    for res, req in result_files:
        name = res.split('/')[-1].split('.')[0]
        result = json.load(open(res))
        request = json.load(open(req))
        size = len(result)
        train_acc = check_result(result[:int(size*0.8)], request[:int(size*0.8)])
        test_acc = check_result(result[int(size*0.8):], request[int(size*0.8):])
        status[name] = {
            'train_set':result[:int(size*0.8)],
            'test_set':result[int(size*0.8):],
            'train_score': train_acc[1],
            'test_score': test_acc[1], 
            'train_target_option': train_acc[2],
            'test_target_option': test_acc[2],
        }
        print(f"{name} | Train Acc: {train_acc[0]} | Test Acc: {test_acc[0]}" )    
    return status


def semantic_retrieval(query, corpus, top_k=1, index=None, device='cuda'):
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = corpus_embeddings.to(device)
    corpus_embeddings = normalize_embeddings(corpus_embeddings)

    query_embeddings = embedder.encode(query, convert_to_tensor=True)
    query_embeddings = query_embeddings.to(device)
    query_embeddings = normalize_embeddings(query_embeddings)
    hits = semantic_search(query_embeddings, corpus_embeddings, score_function=dot_score, top_k=top_k)
    return hits


def post_feedback(feedback):
    pattern = r"choose the '.*' option."
    new_feedback = []
    for x in feedback:
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except:
                x = x
        if not isinstance(x, dict) or "guideline" not in x.keys():
            print(x)
            x = {'reason': x, 'guideline': ''}
        if re.findall(pattern, x['guideline']):
            print(re.findall(pattern, x['guideline']))
        x['guideline'] = re.sub(pattern, "choose the option which doesn't make a decision.", x['guideline'])
        new_feedback.append(x)
    return new_feedback

def build_dataset(prompt, target):
    dataset = []
    for x, y in zip(prompt, target):
        ins = {'input': x, 'target': y}
        dataset.append(ins)
    return dataset