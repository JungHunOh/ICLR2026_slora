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

for seed in [1,2,3]:
    for r in [8,32,128]:
        for dl, bs, epoch in [(32,8,10),(32,8,5),(100,8,5)]:
        #for dl, bs, epoch in [(500,16,5),(500,8,5)]:
            for lr in [2e-4, 1e-4]:
                # os.system(f'CUDA_VISIBLE_DEVICES={gpu} python3 -m torch.distributed.launch --master_addr localhost --master_port 1231 --nproc_per_node=4 --use_env train_math.py \
                #     --model_name_or_path {base_model}\
                #     --data_path ft-training_set/MetaMathQA-40K.json \
                #     --data_length 10000000 \
                #     --bf16 True \
                #     --output_dir ./trained_models/{model}_metamath_slora_r{r}_lr{lr}_seed{seed}/\
                #     --per_device_train_batch_size 8 \
                #     --per_device_eval_batch_size 4 \
                #     --gradient_accumulation_steps 4 \
                #     --evaluation_strategy "no" \
                #     --save_strategy "no" \
                #     --learning_rate {lr}\
                #     --weight_decay 0. \
                #     --warmup_ratio 0.03 \
                #     --logging_steps 1 \
                #     --num_train_epochs 5 \
                #     --lr_scheduler_type "cosine"\
                #     --target_modules q_proj k_proj v_proj up_proj down_proj \
                #     --lora_r {r}\
                #     --lora_alpha {r*2}\
                #     --seed {seed}\
                #     --lora_dropout 0\
                #     --sign_preserve\
                #     ')
                
                os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                    --model_name_or_path {base_model}\
                    --data_path ft-training_set/MetaMathQA-40K.json \
                    --data_length {dl} \
                    --bf16 True \
                    --output_dir ./trained_models/{model}_metamath{dl}bs{bs}epoch{epoch}_slora_r{r}_lr{lr}_seed{seed}/\
                    --per_device_train_batch_size {bs} \
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
                    --lora_alpha {r*2}\
                    --seed {seed}\
                    --lora_dropout 0\
                    --sign_preserve\
                    ')

                os.system(f'CUDA_VISIBLE_DEVICES={gpu} python eval_gsm8k.py --model ./trained_models/{model}_metamath{dl}bs{bs}epoch{epoch}_slora_r{r}_lr{lr}_seed{seed}/ --data_file ./dataset/GSM8K_test.jsonl')
                #os.system(f'python eval_math.py --model ./trained_models/{model}_metamath_lora_r{r}_lr{lr}_seed{seed}/ --data_file ./dataset/MATH_test.jsonl')
