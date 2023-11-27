
import os
import re
import json
import torch

from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from manifest import Manifest
from utils.utils import batchify
from utils.model_utils import postprocess_output, gpt3, codex, T5, LLAMA
    
class llmCollator():
    def __init__(self, tokenizer, target_tokenizer = None):
        self.tokenizer = tokenizer
        if target_tokenizer is None:
            self.target_tokenizer = tokenizer
        else:
            self.target_tokenizer = target_tokenizer

    def __call__(self, instance):
        source = [x['input'] for x in instance]
        target = [x['target'] for x in instance]

        source = self.tokenizer(source, return_tensors="pt", padding=True)
        source_id, source_mask = source['input_ids'], source['attention_mask']
        target = self.target_tokenizer(target, return_tensors="pt", padding=True)
        target_id, target_mask = target['input_ids'], target['attention_mask']

        return source_id, source_mask, target_id, target_mask 


class LLM():
    def __init__(self, model_name, max_len=512, config=None, device="cpu"):
        self.device = device
        self.config = config
        self.model_name = model_name
        self.max_len = max_len

        if model_name in ["text-davinci", "text-davinci-001", "text-davinci-002", "text-davinci-003"]:
            self._model = gpt3
        elif model_name in ["code-davinci-002", "code-davinci-001"]:
            self._model = codex
        elif model_name == "google/flan-t5-xxl":
            self._model = T5(model_name, device=device)
        elif model_name == "yahma/llama-7b-hf":
            self._model = LLAMA(model_name, device=device)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self._target_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._target_tokenizer.pad_token = self._target_tokenizer.pad_token
            self._target_tokenizer.pad_token_id = self._target_tokenizer.eos_token_id
            self.config['pad_token_id'] = self._tokenizer.eos_token_id
            self._model = self._model.eval()

        self.dataset = None
    
    def get_dataloader(self, dataset, batch_size, shuffle=False):
        collator = llmCollator(tokenizer = self._tokenizer, target_tokenizer=self._target_tokenizer)
        dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collator)
        return dataloader

    def request(self, dataset, batch_size, stop_string=None, output_regex=None, remove_str=None):
        generation_config = self.config

        prompts = [x['input'] for x in dataset]
        response = self._model(prompts, max_len=self.max_len, batch_size=batch_size, **generation_config)

        clear_response = [postprocess_output(r, max_length=None, stop_string = stop_string, output_regex = output_regex) for r in response]
        if remove_str:
            clear_response = [x.replace(remove_str, '') for x in clear_response]

        results = []
        for d, out_ori, out_str in zip(dataset, response, clear_response):
            result = {
                "response/prompt_str":  d['input'],
                "response/target_str":  d['target'],
                "response/output_ori": out_ori,
                "response/output": out_str,
            }
            results.append(result)
        return results

                
    def inference(self, dataset, batch_size, shuffle=False, stop_string=None, output_regex=None, remove_str=None):
        dataloader = self.get_dataloader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        generation_config = self.config
        
        response = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            prompt_ids, prompt_mask, target_ids, target_mask = batch

            batch_size, prompt_len = prompt_ids.size()
            prompt_ids, prompt_mask = prompt_ids.to(self.device), prompt_mask.to(self.device)
            target_ids, target_mask = target_ids.to(self.device), target_mask.to(self.device)
            
            outputs = self._model.generate(prompt_ids, attention_mask=prompt_mask, **generation_config)
            outputs = outputs[:,prompt_len:]
            output_ori, output_str = self.post_process(outputs, stop_string=stop_string, output_regex=output_regex, remove_str=remove_str)

            batch_result = []
            for x, y, out, out_ori, out_str in zip(prompt_ids, target_ids, outputs, output_ori, output_str):
                result = {
                    "response/prompt_str":  self.decode(x),
                    "response/target_str":  self.decode(y),
                    "response/output_ori": out_ori,
                    "response/output": out_str,
                }
                batch_result.append(result)

            response.extend(batch_result)
        
        return response

    def decode(self, text):
        text_str = self._tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
        return text_str

    def post_process(self, outputs, max_length = None, stop_string=None, output_regex=None, remove_str = None):
        response = [self.decode(x) for x in outputs]
        clear_response = [postprocess_output(r, max_length=max_length, stop_string = stop_string, output_regex = output_regex) for r in response]
        if remove_str:
            clear_response = [x.replace(remove_str, '') for x in clear_response]
        return response, clear_response

    def generate(self, dataset, batch_size, shuffle=False, stop_string=None, output_regex=None, remove_str=None):
        if self.model_name.startswith("gpt2"):
            return self.inference(dataset, batch_size, shuffle, stop_string, output_regex, remove_str)
        else:
            return self.request(dataset, batch_size, stop_string, output_regex, remove_str)


# python3 -m manifest.api.app     --model_type huggingface     --model_name_or_path EleutherAI/gpt-neox-20b     --device 4
class ManifestLLM(LLM):
    def __init__(self, model_name, client_name, client_connection, engine=None, config=None):
        self.config = config
        self.model_name = model_name
        if engine:
            self._manifest = Manifest(
                    client_name = client_name,
                    engine = engine,
                    client_connection = client_connection,
                )
        else:
            self._manifest = Manifest(
                    client_name = client_name,
                    client_connection = client_connection,
                )
        print(self._manifest.client.get_model_params())
    
    def post_process(self, response, max_length = None, stop_string=None, output_regex=None, remove_str = None):
        clear_response = [postprocess_output(r, max_length=max_length, stop_string = stop_string, output_regex = output_regex) for r in response]
        if remove_str:
            clear_response = [x.replace(remove_str, '') for x in clear_response]
        return clear_response
    
    def inference(self, dataset, batch_size, shuffle=False, stop_string=None, output_regex=None, remove_str=None):
        generation_config = self.config
        if "max_new_tokens" in generation_config:
            generation_config['max_tokens'] = generation_config['max_new_tokens']
            # generation_config["max_length"] = generation_config["max_new_tokens"] + generation_config["length"]
            generation_config.pop("max_new_tokens")
        
        response = []
        for batch in tqdm(batchify(dataset, batch_size), total=len(dataset)//batch_size+1):
            prompt = [x['input'] for x in batch]
            target = [x['target'] for x in batch]

            outputs = self._manifest.run(prompt, **generation_config)
            clean_output = self.post_process(outputs, stop_string=stop_string, output_regex=output_regex, remove_str=remove_str)

            batch_result = []
            for p, t, out_ori, out_str in zip(prompt, target, outputs, clean_output):
                result = {
                    "response/prompt_str":  p,
                    "response/target_str":  t,
                    "response/output_ori": out_ori,
                    "response/output": out_str,
                }
                batch_result.append(result)

            response.extend(batch_result)
        
        return response