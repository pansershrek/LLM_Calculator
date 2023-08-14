import os

import random
import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import pipeline, AutoTokenizer, BloomForCausalLM
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from prompter import Prompter


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MICRO_BATCH_SIZE = 8
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "query_key_value"
]
DATA_PATH = "dataset/bloom-3b-add-dataset.json"
cutoff_len = 512
train_on_inputs = False
model_name = "bigscience/bloom-560m"
setup_seed(42)

model = BloomForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom')

tokenizer.pad_token_id = 0

tokenizer.padding_side = "left"

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0
data = load_dataset("json", data_files=DATA_PATH)

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

prompter = Prompter()

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
val_data = val_data.shuffle().map(generate_and_tokenize_prompt)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir="lora_weights",
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters= None,
        group_by_length=False
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint = False)

model.save_pretrained("lora_weights")

print("\n Finish fine-tuning")
