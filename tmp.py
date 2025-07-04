import torch
import sys

import os

import matplotlib.pyplot as plt

import numpy as np

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_math import smart_tokenizer_and_embedding_resize

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = input()

r = 1
pr = 0

#name = f'trained_models/llama3_commonsense_170k_dl1000bs32epoch5_lora_r{r}_lr0.0001_seed1'
name = f'trained_models/llama3_metamath500bs32epoch5_lora_r{r}_lr0.0001_seed1'

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

tokenizer = AutoTokenizer.from_pretrained(name)

i = 0
ortho = 0
align = 0
num = 0
origin = 0
used = 0
for m in model.modules():
    if hasattr(m, 'lora_A'):
        i += 1
        scale = m.scaling['default']
        weight = m.base_layer.weight.to(torch.float32)
        a = m.lora_A['default'].weight.clone()
        b = m.lora_B['default'].weight.clone()

        # ba = b@a
        # weight = weight.data + ba * scale
        # m.base_layer.weight.data = weight.to(torch.float16)
        torch.nn.init.zeros_(m.lora_A['default'].weight)
        torch.nn.init.zeros_(m.lora_B['default'].weight)
        
        #origin += r * (weight.shape[0]+weight.shape[1])

        
        # ba = b@a
        # btb = b.T @ b
        # aat = a @ a.T
        # I = torch.eye(btb.shape[0], device=btb.device, dtype=btb.dtype)
        # ortho += torch.norm(btb - I, p='fro') ** 2 + torch.norm(aat - I, p='fro') ** 2
        
        #norm_A = torch.norm(a, p='fro')
        #norm_B = torch.norm(b, p='fro')
        #norm_BA = torch.norm(b@a, p='fro')
        #align += norm_A * norm_B / norm_BA

        #num += 1
        
        # #print(torch.norm(btb - I, p='fro'), torch.norm(aat - I, p='fro'))
        #print(f'a: {torch.norm(a, p='fro')}, b: {torch.norm(b, p='fro')}, |a||b|: {torch.norm(a, p='fro')*torch.norm(b, p='fro')} , b@a: {torch.norm(b@a, p='fro')}')
        '''
        U, S, Vh = torch.linalg.svd(weight.data, full_matrices=True)
        C = U.T @ b @ a @ Vh.T

        c_tmp = C.clone()
        
        # #if i % 6 == 0:
        # if False:
        #     if not os.path.exists(f'plt/{i}_pr{pr}_r{r}.png'):
        #         flat_indices = torch.topk(abs(c_tmp).flatten(), int(C.numel()*(1-pr))).indices
        #         rows, cols = flat_indices // c_tmp.shape[1], flat_indices % c_tmp.shape[1]
                
        #         plt.cla()
        #         plt.clf()
        #         plt.imshow(c_tmp.detach().cpu(), cmap='viridis')
        #         plt.scatter(cols.detach().cpu(), rows.detach().cpu(), color='red', marker='o', label='Top-k values', s=0.1)
        #         plt.gca().invert_yaxis()
        #         plt.tight_layout()
        #         plt.savefig(f'plt/{i}_pr{pr}_r{r}.png')
        
        topk = abs(C).view(-1).topk(int(C.numel()*(1-pr)))[0][-1]
        C = C * (abs(C) >= topk)

        # #weight = weight.data + (b@a) * scale#(U @ C @ Vh).clone().detach() * scale
        weight = weight.data + (U @ C @ Vh).clone().detach() * scale

        m.base_layer.weight.data = weight.to(torch.float16)
        torch.nn.init.zeros_(m.lora_A['default'].weight)
        torch.nn.init.zeros_(m.lora_B['default'].weight)
        '''
        

#print(f'ortho: {ortho.item()/ num}, align: {align.item()/ num}')
#print(f'align: {align.item()/ num}')
#print(f'pr: {1-used/origin}')
#model = model.merge_and_unload()

#model.save_pretrained('trained_models/modified_c')


# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
# tokenizer.padding_side = "left"
# tokenizer.pad_token_id = (
#     0  # unk. we want this to be different from the eos token
# )

# model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
# model.config.bos_token_id = 1
# model.config.eos_token_id = 2


if 'metamath' in name:
    from eval_gsm8k import gsm8k_test_noargs
    gsm8k_test_noargs(model, tokenizer, 'modified_c')
else:
    from commonsense_evaluate_func import eval
    eval(model, tokenizer, 'modified_c')

print(f'r{r}_pr{pr}')
input()

'''
for i in [1,5,9,13,17,21]:
    q_a = model.model.model.layers[i].self_attn.q_proj.lora_A['default'].weight
    q_b = model.model.model.layers[i].self_attn.q_proj.lora_B['default'].weight

    U, S, Vh = torch.linalg.svd(model.model.model.layers[i].self_attn.q_proj.base_layer.weight.to(torch.float32), full_matrices=True)
    m, n = U.shape[0], Vh.shape[0]
    S_tmp = torch.zeros(m,n).cuda()
    S_tmp[torch.arange(S.shape[0]),torch.arange(S.shape[0])] = S
    S = S_tmp

    C = U.T @ q_b @ q_a @ Vh.T

    mse = (U @ S @ Vh - U @ (S+C) @ Vh).pow(2).mean() / (U @ S @ Vh).pow(2).mean()
    mse = round(mse.item(),5)

    C = C.detach().cpu().numpy()
    
    plt.cla()
    plt.clf()
    #plt.imshow(abs(C), cmap='gray')
    plt.hist(C.reshape(-1), bins=1000)
    plt.xlim(-0.003,0.003)
    #plt.savefig(f'./plt/{i}_q_mse{mse}.png')
    plt.savefig(f'./plt/hist/{i}_r{r}_wd_q_hist.png')

    k_a = model.model.model.layers[i].self_attn.k_proj.lora_A['default'].weight
    k_b = model.model.model.layers[i].self_attn.k_proj.lora_B['default'].weight

    U, S, Vh = torch.linalg.svd(model.model.model.layers[i].self_attn.k_proj.base_layer.weight.to(torch.float32), full_matrices=True)
    m, n = U.shape[0], Vh.shape[0]
    S_tmp = torch.zeros(m,n).cuda()
    S_tmp[torch.arange(S.shape[0]),torch.arange(S.shape[0])] = S
    S = S_tmp

    C = U.T @ k_b @ k_a @ Vh.T

    mse = (U @ S @ Vh - U @ (S+C) @ Vh).pow(2).mean() / (U @ S @ Vh).pow(2).mean()
    mse = round(mse.item(),5)

    C = C.detach().cpu().numpy()
    
    plt.cla()
    plt.clf()
    #plt.imshow(abs(C), cmap='gray')
    plt.hist(C.reshape(-1), bins=1000)
    plt.xlim(-0.003,0.003)
    #plt.savefig(f'./plt/{i}_k_mse{mse}.png')
    plt.savefig(f'./plt/hist/{i}_r{r}_wd_k_hist.png')

    v_a = model.model.model.layers[i].self_attn.v_proj.lora_A['default'].weight
    v_b = model.model.model.layers[i].self_attn.v_proj.lora_B['default'].weight

    U, S, Vh = torch.linalg.svd(model.model.model.layers[i].self_attn.v_proj.base_layer.weight.to(torch.float32), full_matrices=True)
    m, n = U.shape[0], Vh.shape[0]
    S_tmp = torch.zeros(m,n).cuda()
    S_tmp[torch.arange(S.shape[0]),torch.arange(S.shape[0])] = S
    S = S_tmp

    C = U.T @ v_b @ v_a @ Vh.T

    mse = (U @ S @ Vh - U @ (S+C) @ Vh).pow(2).mean() / (U @ S @ Vh).pow(2).mean()
    mse = round(mse.item(),5)

    C = C.detach().cpu().numpy()
    
    plt.cla()
    plt.clf()
    #plt.imshow(abs(C), cmap='gray')
    plt.hist(C.reshape(-1), bins=1000)
    plt.xlim(-0.003,0.003)
    #plt.savefig(f'./plt/{i}_v_mse{mse}.png')
    plt.savefig(f'./plt/hist/{i}_r{r}_wd_v_hist.png')
'''