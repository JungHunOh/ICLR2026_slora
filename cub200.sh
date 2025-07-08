export CUDA_VISIBLE_DEVICES=$1

for r in 1 128 256 512 1024; do
    dataset=resisc45
    python train_cub200.py \
    --eval_strategy epoch \
    --save_strategy no \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --label_names labels \
    --remove_unused_columns False \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 256 \
    --seed 42 \
    --num_train_epochs 10 \
    --lora_rank $r \
    --output_dir ./cub200_results/${dataset}_lora_r${r}/ \
    --model_name vit-base \
    --finetuning_method lora \
    --dataset_name $dataset \
    --clf_learning_rate 4e-3 \
    --other_learning_rate 4e-3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \

done


python train_cub200.py \
    --eval_strategy epoch \
    --save_strategy no \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --label_names labels \
    --remove_unused_columns False \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 256 \
    --seed 42 \
    --num_train_epochs 100 \
    --lora_rank $r \
    --output_dir ./cub200_results/${dataset}_lora_r${r}_epoch100/ \
    --model_name vit-base \
    --finetuning_method lora \
    --dataset_name $dataset \
    --clf_learning_rate 4e-3 \
    --other_learning_rate 4e-3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \