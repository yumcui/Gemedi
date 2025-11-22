import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os
# --- 1. config model and data paths ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NEW_MODEL_NAME = "llama3-synthetic-patient-generator"

data_files = {
    "train": "phi_training_alpaca_train.jsonl",
    "validation": "phi_training_alpaca_val.jsonl"
}

# --- 2. load data ---
dataset = load_dataset("json", data_files=data_files)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_compute_dtype=torch.float16,# 4-bit for storage, float16 for computation
    bnb_4bit_use_double_quant=False, # use double quantization
)

# --- 5. load model and tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"# auto map to available devices
)
model.config.use_cache = False # set to False to enable gradient checkpointing
model.config.pretraining_tp = 1 # set to 1 to avoid warnings


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # set padding token
tokenizer.padding_side = "right"

# --- 6. Configure LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,# scaling factor, usually between 16 and 32, means how much importance to give to the LoRA layers
    lora_dropout=0.1,# Avoid overfitting
    r=64, # rank of the update matrices
    bias="none",
    task_type="CAUSAL_LM",
    #fine-tune fully connected layers of the transformer
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 7. Define formatting function ---

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{example['instruction'][i]}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output'][i]}<|eot_id|>"""
        output_texts.append(text)
    return output_texts

# --- 8. training arguments and trainer ---
training_arguments = TrainingArguments(
    output_dir="./results_generator",
    num_train_epochs=3,            
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=4, # accumulate gradients， simulating a larger batch size
    optim="paged_adamw_32bit",# avoid overflow when training with fp16
    save_strategy="steps",
    save_steps=50,                
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    gradient_checkpointing=True,# enable gradient checkpointing to save memory, compute gradients rather than storing activations
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    eval_strategy="steps",  
    eval_steps=50                 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,           
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)