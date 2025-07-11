import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

import matplotlib.pyplot as plt

import numpy as np

import random

from train_math import smart_tokenizer_and_embedding_resize

os.environ['CUDA_VISIBLE_DEVICES'] = input()

r = 1
pr = 0

#name = f'trained_models/llama3_commonsense_170k_dl1000bs32epoch5_lora_r{r}_lr0.0001_seed1'
name = f'trained_models/llama3_metamath500000bs32epoch3_lora_r1_lr0.0001_seed1'
name2 = f'trained_models/llama3_metamath500000bs32epoch3_lora_r128_lr0.0001_seed1'

tokenizer = AutoTokenizer.from_pretrained(name)

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B',
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
) # fix zwq

if 'metamath' in name:
    DEFAULT_PAD_TOKEN = "[PAD]"
    smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

model = PeftModel.from_pretrained(
    model,
    name,
    torch_dtype=torch.float16,
    device_map={"":0}
)

model2 = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B',
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
) # fix zwq

if 'metamath' in name:
    DEFAULT_PAD_TOKEN = "[PAD]"
    smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model2,
            )

model2 = PeftModel.from_pretrained(
    model2,
    name2,
    torch_dtype=torch.float16,
    device_map={"":0}
)


tokenizer = AutoTokenizer.from_pretrained(name)

i = 0
sim = 0
num = 0
p = 0.5

for m,m2 in zip(model.modules(), model2.modules()):
    if hasattr(m, 'lora_A'):
        i += 1
        scale = m.scaling['default']
        weight = m.base_layer.weight.to(torch.float32)
        a = m.lora_A['default'].weight.clone()
        b = m.lora_B['default'].weight.clone()

        scale2 = m2.scaling['default']
        a2 = m2.lora_A['default'].weight.clone()
        b2 = m2.lora_B['default'].weight.clone()

        ba = b@a
        ba2 = b2@a2
        print(torch.norm(ba*scale), torch.norm(ba2*scale2), torch.norm(ba*scale-ba2*scale2, p='fro'))
        p = random.uniform(0, 1)
        weight = weight.data + (ba*scale*p + ba2*scale2*(1-p))
        m.base_layer.weight.data = weight.to(torch.float16)
        torch.nn.init.zeros_(m.lora_A['default'].weight)
        torch.nn.init.zeros_(m.lora_B['default'].weight)
        
if 'metamath' in name:
    from eval_gsm8k import gsm8k_test_noargs
    gsm8k_test_noargs(model, tokenizer, 'modified_c', end=300)
else:
    from commonsense_evaluate_func import eval
    eval(model, tokenizer, 'modified_c')

print(f'r{r}_pr{pr}')
input()
