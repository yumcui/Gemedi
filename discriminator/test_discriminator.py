import json
import csv
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./llama3-discriminator"
EVAL_DATA_PATH = "eval_discriminator.jsonl"
OUTPUT_CSV = "evaluation_results_full.csv"

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


def load_model_and_tokenizer():
    print(f"Loading base model: {BASE_MODEL_ID} ...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading LoRA adapter: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    return model, tokenizer


def generate_prediction(model, tokenizer, note_content):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Medical Note:\n---\n{note_content}\n---"},
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def main():

    model, tokenizer = load_model_and_tokenizer()

    print(f"Reading data from {EVAL_DATA_PATH}...")
    lines = []
    try:
        with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Error: File {EVAL_DATA_PATH} not found.")
        return

    print(f"Writing results to {OUTPUT_CSV}...")

    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)

        writer.writerow(["original_note", "original_reason", "predicted_result"])

        for line in tqdm(lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)

                note_content = item.get("note", item.get("input", item.get("content", "")))
                gt_reason = item.get("reason", item.get("output", item.get("label", "")))

                if not note_content:
                    continue

                prediction = generate_prediction(model, tokenizer, note_content)

                writer.writerow([note_content, gt_reason, prediction])

                f_out.flush() 

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line.")
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"\nDone! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()