r=1
dataset=resisc45
export CUDA_VISIBLE_DEVICES=$1
python vit_analysis.py \
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
  --lora_rank 1 \
  --output_dir ./cub200_results/test/ \
  --model_name vit-base \
  --finetuning_method lora \
  --dataset_name $dataset \
  --clf_learning_rate 4e-3 \
  --other_learning_rate 4e-3 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \


