import os

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

#model='gpt'
#model='llama'
model='llama3'

lr=5e-4
r=32
seed=1

if model == 'gpt':
    base_model = 'EleutherAI/gpt-j-6b'
elif model == 'llama':
    base_model = 'yahma/llama-7b-hf'
elif model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'

lr = 1e-4
for seed in [1]:
    for r, target_r, alpha, epoch_p in [(1,4,8,0.1),(1,4,16,0.1),(1,4,16,0.2),(1,64,16,0.2),(1,64,16,0.3),(1,128,16,0.4)]:
        for dl, bs, epoch in [(1000000,32,5)]:
            os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                --model_name_or_path {base_model}\
                --data_path ft-training_set/MetaMathQA-40K.json \
                --data_length 10000000 \
                --bf16 True \
                --output_dir ./trained_models/{model}_metamath{dl}bs{bs}epoch{epoch}_lora_r{r}_target_r{target_r}_alpha{alpha}_lr{lr}_epoch{epoch_p}_seed{seed}/\
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --evaluation_strategy "no" \
                --save_strategy "no" \
                --learning_rate {lr}\
                --weight_decay 0. \
                --warmup_ratio 0.03 \
                --logging_steps 1 \
                --num_train_epochs {epoch} \
                --lr_scheduler_type "cosine"\
                --target_modules q_proj k_proj v_proj up_proj down_proj \
                --lora_r {r}\
                --lora_alpha {alpha}\
                --seed {seed}\
                --lora_dropout 0\
                --epoch_p {epoch_p}\
                --target_r {target_r}\
                ')

            #os.system(f'CUDA_VISIBLE_DEVICES={gpu} python eval_gsm8k.py --model ./trained_models/{model}_metamath_lora_r{r}_lr{lr}_seed{seed}/ --data_file ./dataset/GSM8K_test.jsonl')
            #os.system(f'python eval_math.py --model ./trained_models/{model}_metamath_lora_r{r}_lr{lr}_seed{seed}/ --data_file ./dataset/MATH_test.jsonl')
