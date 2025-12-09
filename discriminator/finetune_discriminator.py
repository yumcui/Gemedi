import os
import argparse
import json
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


SYSTEM_PROMPT = """You are an expert clinical documentation auditor specializing in detecting:
- factual inconsistencies
- missing clinical justification
- hallucinated diagnoses or treatments
- contradictory or impossible temporal sequences
- malformed or contradictory PHI

Your task is to analyze the medical note and identify *any* of the following errors:
- Internal Clinical Logic Errors
- Timeline Hallucinations
- PHI Inconsistencies
- Structural / Format Violations
- Common Hallucination Patterns

OUTPUT FORMAT:
Return ONLY this JSON object:
{
  "reason": "good if no issues, otherwise describe the error in ≤50 words"
}

EXAMPLES:
✓ Good note → {"reason": "good"}
✗ Bad note → {"reason": "Name error:'Wagner'[less than 2 words]; admission date >= discharge date"}"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--train_data", type=str, default="train_discriminator.jsonl")
    parser.add_argument("--eval_data", type=str, default="eval_discriminator.jsonl")
    parser.add_argument("--output_dir", type=str, default="./llama3-discriminator")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------
    # Load 4-bit QLoRA base model
    # -----------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # -----------------------
    # LoRA config
    # -----------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # -----------------------
    # Data collator
    # -----------------------
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # -----------------------
    # Formatting function
    # -----------------------
    def formatting_prompts_func(examples):
        outputs = []
        for note, reason in zip(examples["note"], examples["reason"]):

            clean_reason = reason.strip() or "good"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Medical Note:\n---\n{note}\n---"},
                {"role": "assistant", "content": json.dumps({"reason": clean_reason}, ensure_ascii=False)},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            outputs.append(text)

        return outputs

    # -----------------------
    # Datasets
    # -----------------------
    train_dataset = load_dataset("json", data_files={"train": args.train_data}, split="train")
    eval_dataset = load_dataset("json", data_files={"eval": args.eval_data}, split="eval")

    # -----------------------
    # Training arguments
    # -----------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,

        bf16=True,
        gradient_checkpointing=True,
        group_by_length=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
