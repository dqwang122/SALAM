from pathlib import Path
import yaml

NEGATIVE_INF = -100000.0
OPENAI_API_KEY=""

STOPSTRING = "\n"
REMOVESTR = "\n"

LOCALHOST = "http://127.0.0.1:5000"

MODELNAME = {
    "gpt2_xl": "gpt2-xl",
    "gpt2_neox": "EleutherAI/gpt-neox-20b",
    "flan_t5": "google/flan-t5-xxl", 
    "gpt3": "text-davinci",
    "instructgpt": "text-davinci-003", 
    "codex": "code-davinci-002",
    "chatgpt": "gpt-3.5-turbo",
    "llama": "yahma/llama-7b-hf",
    "llama_big": "yahma/llama-13b-hf",
    "vicuna": "eachadea/vicuna-7b-1.1",
    "vicuna_big":"eachadea/vicuna-13b-1.1"
}


TEMPLATE = """{question}
We get the answer {ans} from the model while the true answer is {target}. 
Please return a json with the following keys:
    reason: explain the potential reason of the model prediction
    guideline: based on the reason, provide an instruction as a prompt for the model to avoid similar mistakes
Please do not mention the true answer or any specific option content in your response."""

TEMPLATE_TEST = """{question}
Please return a json with the following keys:
    reason: explain the potential reason of the model potential prediction
    guideline: based on the reason, provide an instruction as a prompt for the model to avoid similar mistakes
Please do not mention the true answer or any specific option content in your response."""


GREEDY_CONFIG = {
    "max_new_tokens": 50,
    "do_sample": False,
    "temperature": 0,
    "num_beams": 5,
}

COT_GREEDY_CONFIG = {
    "max_new_tokens": 100,
    "do_sample": False,
    "num_beams": 5,
}

SAMPLE_CONFIG = {
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1
}

COT_SAMPLE_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1
}

