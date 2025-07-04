import os

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

for seed in [1]:
    for r in [1,128]:
        for dl, bs, epoch in [(500000,32,5)]:
            for lr in [1e-4]:
                os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                    --model_name_or_path {base_model}\
                    --data_path ft-training_set/MetaMathQA-40K.json \
                    --data_length {dl} \
                    --bf16 True \
                    --output_dir ./trained_models/{model}_metamath{dl}bs{bs}epoch{epoch}_lora_r{r}_lr{lr}_seed{seed}/\
                    --per_device_train_batch_size 8 \
                    --per_device_eval_batch_size 4 \
                    --gradient_accumulation_steps {bs // 8} \
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
                    --lora_alpha {r*2}\
                    --seed {seed}\
                    --lora_dropout 0\
                    ')

                #os.system(f'CUDA_VISIBLE_DEVICES={gpu} python eval_gsm8k.py --model ./trained_models/{model}_metamath{dl}bs{bs}epoch{epoch}_lora_r{r}_lr{lr}_seed{seed}/ --data_file ./dataset/GSM8K_test.jsonl')
                #os.system(f'python eval_math.py --model ./trained_models/{model}_metamath_lora_r{r}_lr{lr}_seed{seed}/ --data_file ./dataset/MATH_test.jsonl')
