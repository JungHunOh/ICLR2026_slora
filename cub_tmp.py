import torch
import sys

import os

import matplotlib.pyplot as plt

import numpy as np

from train_cub200 import get_dataset, get_ids_and_labels_from_dataset, compute_metrics, collate_fn, get_transforms, preprocess

from peft import PeftModel
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments


#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = input()
os.environ["WANDB_DISABLED"] = "true"

r = 1
pr = 0

#name = f'trained_models/llama3_commonsense_170k_dl1000bs32epoch5_lora_r{r}_lr0.0001_seed1'
name = f'cub200_results/lora_r1/checkpoint-160'

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

dataset_train, dataset_val, dataset_test = get_dataset('cifar100')
label2id, id2label = get_ids_and_labels_from_dataset(dataset_train)

transform_fn = get_transforms(image_processor)

dataset_train.set_transform(lambda x: preprocess(x, transform_fn))
dataset_val.set_transform(lambda x: preprocess(x, transform_fn))
dataset_test.set_transform(lambda x: preprocess(x, transform_fn))

model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to('cuda')

model = PeftModel.from_pretrained(
    model,
    name,
)

args = TrainingArguments(
    label_names=["labels"],
    remove_unused_columns=False,
    report_to=None
)

trainer = Trainer(
        model,
        args,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )


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
        weight = m.base_layer.weight
        a = m.lora_A['default'].weight.clone()
        b = m.lora_B['default'].weight.clone()

        ba = b@a
        weight = weight.data + ba * scale
        m.base_layer.weight.data = weight
        torch.nn.init.zeros_(m.lora_A['default'].weight)
        torch.nn.init.zeros_(m.lora_B['default'].weight)
        
        #origin += r * (weight.shape[0]+weight.shape[1])

        
eval_results = trainer.evaluate(dataset_test)
print(eval_results)