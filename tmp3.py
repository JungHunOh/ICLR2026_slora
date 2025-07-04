import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

import matplotlib.pyplot as plt

import numpy as np

import random

os.environ['CUDA_VISIBLE_DEVICES'] = input()

r = 1
pr = 0

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B',
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
) # fix zwq
model = PeftModel.from_pretrained(
    model,
    f'trained_models/llama3_commonsense_170k_dl1000000bs32epoch3_lora_r32_lr0.0001_seed1',
    torch_dtype=torch.float16,
    device_map={"":0}
)

model2 = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B',
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
) # fix zwq
model2 = PeftModel.from_pretrained(
    model2,
    f'trained_models/llama3_commonsense_170k_dl1000000bs32epoch3_lora_r128_lr0.0001_seed1',
    torch_dtype=torch.float16,
    device_map={"":0}
)

i = 0
sim = 0
num = 0
p = 0.2

for m,m2 in zip(model.modules(), model2.modules()):
    if hasattr(m, 'lora_A'):
        i += 1
        scale = m.scaling['default']
        weight = m.base_layer.weight.to(torch.float32)
        a = m.lora_A['default'].weight.clone()
        b = m.lora_B['default'].weight.clone()

        a2 = m2.lora_A['default'].weight.clone()
        b2 = m2.lora_B['default'].weight.clone()

        ba = b@a
        ba2 = b2@a2
        p = random.uniform(0, 1)
        weight = weight.data + (ba*p + ba2*(1-p)) * scale
        m.base_layer.weight.data = weight.to(torch.float16)
        torch.nn.init.zeros_(m.lora_A['default'].weight)
        torch.nn.init.zeros_(m.lora_B['default'].weight)
        
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

from commonsense_evaluate_func import eval
eval(model, tokenizer, 'modified_c')
print(f'r{r}_pr{pr}')
input()
