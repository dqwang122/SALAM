
import os
import re
import json

from tqdm import tqdm
from typing import Iterable, List

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm import LLM, ManifestLLM
from utils.tools import semantic_retrieval, simple_check_result, post_feedback, build_dataset
from utils.constants import MODELNAME, LOCALHOST, STOPSTRING, REMOVESTR, TEMPLATE, TEMPLATE_TEST, SAMPLE_CONFIG


MAX_TARGET_LENGTH = 512



class Memory():
    def __init__(self, max_mem_size, source, memory_dir, prompt_mode='zero_shot',feedback_file=None):
        self.source = source
        self.memory_dir = memory_dir
        self.max_mem_size = max_mem_size
        self.prompt_mode = prompt_mode

        mistake_file = f"{memory_dir}/mistake_collections.json"
        correct_file = f"{memory_dir}/correct_collections.json"
        mistake_collections = json.load(open(mistake_file, 'r'))
        correct_collections = json.load(open(correct_file, 'r'))
        
        print("Correction collections:{} ({})".format(len(correct_collections), correct_file))
        print("Mistake collections:{} ({})".format(len(mistake_collections), mistake_file))

        self.memory = []
        if source == 'none':
            print("Memory built from none")
            self.memory = []
        elif source == 'correct':
            print("Memory built from correct collections")
            for i, x in enumerate(correct_collections):
                key = x['response/prompt_str'] if prompt_mode == 'zero_shot' else self.get_few_shot(x['response/prompt_str'])[1]
                slot = {
                    'indx': i,
                    'key': key,
                    'value': f"{key}.\nThe correct answer is {x['response/target_str']}",
                }
                self.memory.append(slot)
        elif source == 'mistake':
            print("Memory built from mistake collections")
            for i, x in enumerate(mistake_collections):
                key = x['response/prompt_str'] if prompt_mode == 'zero_shot' else self.get_few_shot(x['response/prompt_str'])[1]
                slot = {
                    'indx': i,
                    'key': key,
                    'value': f"{key}.\nPrevious wrong answer is {x['response/output']}. The correct answer is {x['response/target_str']}.",
                }
                self.memory.append(slot)
        elif source == 'feedback':
            feedback = [json.loads(x) for x in open(feedback_file, 'r')]
            feedback = post_feedback(feedback)
            print(f"Memory built from feedback collections ({len(feedback)})")
            for i, (x, fb) in enumerate(zip(mistake_collections, feedback)):
                key = x['response/prompt_str'] if prompt_mode == 'zero_shot' else self.get_few_shot(x['response/prompt_str'])[1]
                slot = {
                    'indx': i,
                    'key': key,
                    'value': f"{key}.\nPrevious wrong answer is {x['response/output']}. The correct answer is {x['response/target_str']}.",
                    'guideline': fb['guideline'],
                    'reason': fb['reason']
                }
                self.memory.append(slot)
        elif source == 'feedback_cot':
            print("Memory built from feedback chain-of-thought collections")
            feedback = [json.loads(x) for x in open(feedback_file, 'r')]
            feedback = post_feedback(self.feedback)
            for i, (x, fb) in enumerate(zip(mistake_collections, feedback)):
                key = x['response/prompt_str'] if prompt_mode == 'zero_shot' else self.get_few_shot(x['response/prompt_str'])[1]
                slot = {
                    'indx': i,
                    'key': key,
                    'value': f"{key}.\nPrevious wrong answer is {x['response/output']}. The correct answer is {x['response/target_str']}.\nReason: {fb['reason']}\nGuideline: {fb['guideline']}",
                    'guideline': fb['guideline'],
                    'reason': fb['reason']
                }
                self.memory.append(slot)
        else:
            self.memory = []
                
        self.memory = self.memory[:max_mem_size]
        

    def retrieve(self, test_query, topk=3, thre=0.9, max_return=20):
        corpus = [slot['key'] for slot in self.memory]
        if self.prompt_mode == 'few_shot':
            test_query = [self.get_few_shot(x)[1] for x in test_query]
        ret_all = semantic_retrieval(test_query, corpus, top_k=max_return)

        retrieval = []
        for i, (q, r) in enumerate(zip(test_query, ret_all)):
            corpus_id = [x['corpus_id'] for x in r if x['score'] > thre]
            corpus_id = corpus_id[:topk]
            retrieved_cases = [self.memory[x]['value'] for x in corpus_id]
            retrieved_content = "\n\n".join(retrieved_cases)
            if self.source == 'feedback':
                retrieved_feedback = [self.memory[x]['guideline'] for x in corpus_id]
                retrieved_content = '\n'.join(retrieved_feedback) + '\n\n' + retrieved_content
            elif self.source == 'feedback_only':
                retrieved_feedback = [self.memory[x]['guideline'] for x in corpus_id]
                retrieved_content = '\n'.join(retrieved_feedback)

            retrieval.append(retrieved_content)
        return retrieval, ret_all

    def update(self, mistake: list, feedback: list, source: str):
        cur_idx = len(self.memory)
        for i, (x, fb) in enumerate(zip(mistake, feedback)):
            if fb['guideline'] == '':
                continue
            
            key = x['response/prompt_str'] if self.prompt_mode == 'zero_shot' else self.get_few_shot(x['response/prompt_str'])[1]

            if source == 'feedback':
                slot = {
                    'indx': i + cur_idx,
                    'key': key,
                    'value': f"{key}.\nPrevious wrong answer is {x['response/output']}. The correct answer is {x['response/target_str']}.",
                    'guideline': fb['guideline'],
                    'reason': fb['reason']

                }
            elif source == 'feedback_cot':
                slot = {
                    'indx': i,
                    'key': key,
                    'value': f"{key}.\nPrevious wrong answer is {x['response/output']}. The correct answer is {x['response/target_str']}. Reason: {fb['reason']}\n Guideline: {fb['guideline']}",
                    'guideline': fb['guideline'],
                    'reason': fb['reason']
                }
            else:
                raise NotImplementedError
            
            self.memory.append(slot)
        
        self.memory = self.memory[-self.max_mem_size:]
        print(f'Current Memory Size is {len(self.memory)})')


    def get_few_shot(self, x):
        examples, query = x.split("\n\n")[:-1], x.split("\n\n")[-1]
        return examples, query

class SALAM():
    def __init__(self, basemodel: str = "flan_t5", study_assistant: str = "yahma/llama-7b-hf", memory_dir='../data/', save_root='../results', device=['cuda:0', 'cuda:1'], llm_config=SAMPLE_CONFIG):
        super().__init__()

        self.llm_name = basemodel
        self.llm_config = llm_config
        self.sa_name = study_assistant
        self.memory_dir = memory_dir
        self.save_root = save_root
        self.device = device
        
        model_name = MODELNAME[basemodel]
        print(f'Building Base Model: {model_name} ({basemodel})')
        if basemodel == "gpt2_neox":
            self.llm = ManifestLLM(model_name=model_name, client_name="huggingface", client_connection=LOCALHOST, config=llm_config)
        else:
            self.llm = LLM(model_name=model_name, config=llm_config, device=device[0])

        print(f"Building Study Assistant {study_assistant}")
        self.sa = StudyAssistant(model_name_or_path=study_assistant, device=device[1])
        self.sa_configs = SAMPLE_CONFIG
        self.sa_configs['max_new_tokens'] = MAX_TARGET_LENGTH

        print(f"Save root: {self.save_root}")
    
    def examine_task(self, name, data, memory, batch_size=1, topk=3, thre=0.9, source='correct', suffix = "The answer is ", save=False):
        test_data, test_query, target, test_target_opt = self.load_data(data)
        retrieval, ret_all = memory.retrieve(test_query, topk=topk, thre=thre)
        new_prompt = []
        for i, (q, r) in enumerate(zip(test_query, retrieval)):
            q = r + "\n\n" + q + '\n' + suffix
            new_prompt.append(q)
            if i == 0:
                print(q)

        response, score = self.response_from_student(new_prompt, target, test_target_opt, batch_size)
        ori_score = round(sum(data['test_score']) * 100 / len(data['test_score']), 2)
        print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )

        if save:
            json.dump(response, open(f"{self.save_root}/{self.llm_name}/{name}_{source}_update_test.json", 'w'), indent=2)

        return response, score

    def examination(self, data, source, max_mem_size, feedback_file, batch_size=1, topk=3, thre=0.9, save=False):
        print(f"Building Memory {self.memory_dir}")
        if self.llm_name.startswith('flan_t5'):
            prompt_mode = "zero_shot"
        else:
            prompt_mode = "few_shot"
        memory = Memory(max_mem_size, source, self.memory_dir, prompt_mode=prompt_mode, feedback_file=feedback_file)

        if source == 'none':
            # orignal test
            for name, v in data.items():
                test_data, test_query, target, test_target_opt = self.load_data(v)
                response, score = self.response_from_student(test_query, target, test_target_opt, batch_size)
                ori_score = round(sum(v['test_score']) * 100 / len(v['test_score']), 2)
                print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )
                if save:
                    json.dump(response, open(f"{self.save_root}/{self.llm_name}/{name}_test.json", 'w'), indent=2)

        elif source == 'correct':
            # retrieve from correct collections
            for name, v in data.items():
                response, score = self.examine_task(name, v, memory, batch_size=batch_size, topk=topk, thre=thre, source='correct', suffix = "The correct answer is ", save=save)
        elif source == 'mistake':
            for name, v in data.items():
                response, score = self.examine_task(name, v, memory, batch_size=batch_size, topk=topk, thre=thre, source='mistake', suffix = "The correct answer is ", save=save)
        elif source == 'feedback':
            for name, v in data.items():
                response, score = self.examine_task(name, v, memory, batch_size=batch_size, topk=topk, thre=thre, source='feedback', suffix = "The correct answer is ", save=save)
        elif source == 'feedback_cot':
            for name, v in data.items():
                response, score = self.examine_task(name, v, memory, batch_size=batch_size, topk=topk, thre=thre, source='feedback_cot', suffix = "The correct answer is ", save=save)
        elif source == 'feedback_only':
            for name, v in data.items():
                response, score = self.examine_task(name, v, memory, batch_size=batch_size, topk=topk, thre=thre, source='feedback_only', suffix = "The correct answer is ", save=save)
        
        elif source == 'agent':
            for name, v in data.items():
                test_data, test_query, target, test_target_opt = self.load_data(v)
                sa_prompt = [TEMPLATE_TEST.format(question=x) for x in test_query]
                sa_ret = self.sa.inference(sa_prompt, batch_size=batch_size, **self.sa_configs)
                feedback = post_feedback(sa_ret)
                new_prompt = []
                for i, (q, r) in enumerate(zip(test_query, feedback)):
                    q = r['guideline'] + '\n'+ q + '\n' + 'The correct answer is '
                    if i == 0:
                        print(q)
                    new_prompt.append(q)

                response, score = self.response_from_student(new_prompt, target, test_target_opt, batch_size)
                ori_score = round(sum(v['test_score']) * 100 / len(v['test_score']), 2)
                print(f"{name} | ori test acc: {ori_score} | new test Acc: {score}" )

                if save:
                    json.dump(response, open(f"{self.save_root}/{self.llm_name}/{name}_{source}_update_test.json", 'w'), indent=2)

        else:
            raise NotImplementedError
    
    def gather(self, data, batch_size=1, max_iteration=4, max_mem_size=1000, topk=3, thre=0.9, save=True):
        memory = Memory(max_mem_size=max_mem_size, source='none', memory_dir=self.memory_dir, feedback_file='None')
        interaction = {name:[] for name in data.keys()}
        for iter in range(max_iteration):
            for name, v in data.items():
                test_data, test_query, target, test_target_opt = self.load_data(v)
                if iter == 0:
                    response, score = self.response_from_student(test_query, target, test_target_opt, batch_size)
                else:
                    response, score = self.examine_task(self, name, data, memory, batch_size=batch_size, topk=topk, thre=thre, source='feedback', suffix = "The answer is ", save=False)

                mistakes = [r for r in response if r['response/score'] == 0]
                sa_prompts = []
                for i, r in enumerate(mistakes):
                    sa_prompt = TEMPLATE.format(question=r['response/prompt_str'], ans=r['response/output'], target=r['response/target_str'])
                    sa_prompts.append(sa_prompt)
                sa_ret = self.sa.inference(sa_prompts, batch_size=batch_size, **self.sa_configs)
                feedback = post_feedback(sa_ret)
                
                memory.update(mistakes, feedback, source='feedback')
                for i, (x, r) in enumerate(mistakes, sa_ret):
                    x['feedback'] = r
                interaction[name].append(x)

        if save:
            json.dump(interaction, open(f"{self.save_root}/{self.llm_name}_interaction.json", 'w'), indent=2)


    def response_from_student(self, test_query, target, test_target_opt, batch_size=1, stop_string=STOPSTRING, remove_str=REMOVESTR):
        dataset = build_dataset(test_query, target)
        response = self.llm.generate(dataset, batch_size, stop_string=stop_string, remove_str=remove_str)
        score, detail = simple_check_result(response, test_target_opt)
        for i, (r, s) in enumerate(zip(response, detail)):
            r['index'] = i
            r['response/score'] = s
        return response, score
    
    def load_data(self, data):
        test_data = data['test_set']
        test_target_opt = data['test_target_option']
        test_query = [x['response/prompt_str'] for x in test_data]
        target = [x['response/target_str'] for x in test_data]
        return test_data, test_query, target, test_target_opt


class StudyAssistant():
    def __init__(self, model_name_or_path: str = "yahma/llama-7b-hf", padding_side: str = "left", device: str = "cuda"):
        super().__init__()
        self.device = device
        
        print(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side=padding_side,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

        print("Vocab Size: ", len(self.tokenizer))
        print("Loaded in model and tokenizer!")
        

        self.model.train()

    def inference(self, prompts: List, batch_size: int = 1, **kwargs) -> Iterable[str]:
        self.model.eval()

        results = []
        chunk_num = len(prompts) // batch_size + (len(prompts) % batch_size > 0)
        scope = tqdm(range(chunk_num)) if chunk_num > 10 else range(chunk_num)
        with torch.no_grad():
            for i in scope:
                batch_x = prompts[i*batch_size:(i+1)*batch_size]
                inputs = self.tokenizer(batch_x, return_tensors="pt", padding=True, truncation=True).to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                outputs = self.model.generate(**inputs, **kwargs)
                outputs = outputs[:,prompt_len:]
                res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(res)
        return results
    
