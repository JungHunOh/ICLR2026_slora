import os

print('enter gpu')
gpu=int(input())

dataset='commonsense_170k'
#dataset='math_10k'

#model='gpt'
#model='llama'
model='llama3'

lr=1e-4
r=32
seed=1

if model == 'gpt':
    base_model = 'EleutherAI/gpt-j-6b'
elif model == 'llama':
    base_model = 'yahma/llama-7b-hf'
elif model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'

# math
os.system(f'CUDA_VISIBLE_DEVICES={gpu} python finetune.py --base_model {base_model} --data_path ./ft-training_set/{dataset}.json --output_dir ./trained_models/{model}_{dataset}_lora_r{r}_lr{lr}_seed{seed}/ --batch_size 16 --micro_batch_size 16   --num_epochs 3   --learning_rate {lr}   --cutoff_len 256   --val_set_size 0 --eval_step 80 --save_step 80  --adapter_name lora --lora_r {r} --lora_alpha {r*2} --seed {seed}')

if dataset == 'commonsense_170k':
    evalsets = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]
    eval_file = 'commonsense_evaluate.py'
else:
    evalsets = ['SVAMP', 'AQuA', 'AddSub', 'gsm8k', 'MultiArith', 'SingleEq']
    eval_file = 'evaluate.py'

for eval_dataset in evalsets:
    if model == 'gpt':
        model_name = 'GPT-j-6B'
    elif model == 'llama':
        model_name = 'LLaMA-7B'
    elif model == 'llama3':
        model_name = 'LLaMA3-8B'
    os.system(f'CUDA_VISIBLE_DEVICES={gpu} python {eval_file} --model {model_name} --adapter LoRA --dataset {eval_dataset} --base_model {base_model} --lora_weights ./trained_models/{model}_{dataset}_lora_r{r}_lr{lr}_seed{seed}')

