import csv
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_JSONL = "eval_discriminator_v2.jsonl"
OUTPUT_CSV = "eval_issues.csv"


# ================================================================
# 1. LOAD MODEL
# ================================================================
def load_llama(model_name):
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ================================================================
# 2. PROMPT TEMPLATE
# ================================================================
PROMPT_TEMPLATE = """
You are an expert clinician specializing in medical consistency checking.

Your tasks:
1. Extract all diagnoses mentioned in the note.
2. Detect inconsistencies between:
    - diagnoses
    - treatments / medications (dose/route/frequency)
    - clinical narrative

STRICT RULES:
- DO NOT modify the content.
- DO NOT rewrite the note.
- Just identify diagnoses & describe any issues.

OUTPUT FORMAT:
Return ONLY this JSON object:

{{
  "diagnoses": [...],
  "issues": "description of inconsistencies"
}}

Here is the medical note:

---
{note}
---
"""


# ================================================================
# 3. INFERENCE
# ================================================================
def run_llama(model, tokenizer, note):
    prompt = PROMPT_TEMPLATE.format(note=note)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    diagnoses = ""
    issues = ""

    # Attempt extracting JSON only
    prompt_text = PROMPT_TEMPLATE.format(note=note)
    if decoded.startswith(prompt_text):
        json_response_text = decoded[len(prompt_text):].strip()
    else:
        json_response_text = decoded.strip()

    json_match = re.search(r"(\{[\s\S]*?\})", json_response_text, re.DOTALL)

    if json_match:
        json_text = json_match.group(0).strip()

        while True:
            try:
                data = json.loads(json_text)
                diagnoses = data.get("diagnoses", "")
                issues = data.get("issues", "")
                break
            except json.JSONDecodeError as e:
                if "Extra data" in str(e):
                    last_brace_index = json_text.rfind('}')
                    if last_brace_index != -1:
                        json_text = json_text[:last_brace_index + 1].strip()
                        continue

                print(f"\n[WARNING] JSONDecodeError: {e}")
                print(f"[ERROR TEXT] Could not parse: {json_text[:200]}...")
                print("[WARNING] JSON parse failed. Using empty fields.\n")
                return "", ""
            except Exception:
                print("\n[WARNING] JSON parse failed (Other Error). Using empty fields.\n")
                return "", ""
    else:
        print("\n[WARNING] JSON structure {} not found. Using empty fields.\n")

    return issues


# ================================================================
# 4. LOAD JSONL
# ================================================================
def load_jsonl(path):
    print(f"Loading JSONL: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records.")
    return records


# ================================================================
# 5. SAVE CSV (ONLY TWO COLUMNS)
# ================================================================
def save_to_csv(records, csv_path):
    print(f"\nSaving CSV → {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_note", "issues"])

        for rec in records:
            writer.writerow([
                rec["original"],
                rec["issues"],
            ])

    print("Done.")


# ================================================================
# 6. MAIN
# ================================================================
if __name__ == "__main__":
    model, tokenizer = load_llama(MODEL_NAME)
    jsonl_records = load_jsonl(INPUT_JSONL)

    output_rows = []

    for i, record in enumerate(jsonl_records):
        note = record.get("note") or record.get("text") or ""

        print(f"\n=== Processing record {i+1}/{len(jsonl_records)} ===")
        issues = run_llama(model, tokenizer, note)

        output_rows.append({
            "original": note.strip(),
            "issues": issues,
        })

    save_to_csv(output_rows, OUTPUT_CSV)

    print("\nAll done! Output CSV:", OUTPUT_CSV)
