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
SYSTEM_PROMPT = """You are an expert clinical documentation auditor specializing in detecting medical logic errors and inconsistencies.

Your task is to analyze the medical note and identify *any* of the following errors:

1. **Clinical Logic Errors**:
   - Age/sex mismatch with diagnosis (e.g., pediatric condition in elderly, male-specific condition in female)
   - Biologically implausible disease combinations
   - Inappropriate diagnosis for patient demographics

2. **Medication Logic Errors**:
   - Medications that don't match the diagnosis
   - Missing standard medications for the diagnosis
   - Inappropriate medication for patient age/condition
   - Dosage/route/frequency inconsistencies

3. **Timeline Errors**:
   - Admission date >= Discharge date
   - Dates that don't make clinical sense
   - Impossible temporal sequences

4. **PHI (Protected Health Information) Inconsistencies**:
   - Name format errors (less than 2 words, invalid characters)
   - Date format inconsistencies
   - Age calculation errors (DOB vs admission/discharge dates)

5. **Internal Consistency Errors**:
   - Contradictions between different sections
   - Discharge instructions that don't match diagnosis
   - Treatment plan inconsistencies

STRICT RULES:
- DO NOT modify or rewrite the note.
- If you find NO errors, output exactly "good".
- If you find errors, provide a concise reasoning (≤50 words) describing the specific issues.
- Do NOT invent problems that don't exist.
- Focus on factual medical logic errors, not stylistic preferences.

OUTPUT FORMAT:
Return ONLY a JSON object with this structure:
{
  "issues": "good" OR "concise description of the medical logic error(s)"
}

EXAMPLES:
✓ Good note → {"issues": "good"}
✗ Age mismatch → {"issues": "Age 5 with diagnosis 'Benign Prostatic Hyperplasia' (BPH) is biologically impossible; BPH only occurs in adult males"}
✗ Timeline error → {"issues": "Admission date (2024-09-15) is after discharge date (2024-09-10)"}
✗ Medication mismatch → {"issues": "Diagnosis is 'Acute Myocardial Infarction' but no antiplatelet or statin medications prescribed"}
"""

# ================================================================
# 3. INFERENCE
# ================================================================
def run_llama(model, tokenizer, note):
    """
    Run inference using chat template format (better for Llama 3.1 Instruct)
    """
    # Build messages using chat format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Medical Note:\n---\n{note}\n---\n\nAnalyze this medical note for logic errors and inconsistencies."}
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,  # Reduced since we only need "good" or short reasoning
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated part (not including the prompt)
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Try to extract JSON from response
    json_match = re.search(r'\{[\s\S]*?"issues"[\s\S]*?\}', decoded, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(0).strip()
        
        # Try to parse JSON
        while True:
            try:
                data = json.loads(json_text)
                issues = data.get("issues", "")
                if issues:
                    return issues
                break
            except json.JSONDecodeError as e:
                if "Extra data" in str(e):
                    last_brace_index = json_text.rfind('}')
                    if last_brace_index != -1:
                        json_text = json_text[:last_brace_index + 1].strip()
                        continue
                
                # If JSON parsing fails, try to extract "issues" value directly
                issues_match = re.search(r'"issues"\s*:\s*"([^"]+)"', json_text)
                if issues_match:
                    return issues_match.group(1)
                
                print(f"\n[WARNING] JSONDecodeError: {e}")
                print(f"[ERROR TEXT] Could not parse: {json_text[:200]}...")
                break
            except Exception as e:
                print(f"\n[WARNING] JSON parse failed: {e}")
                break
    
    # Fallback: check if response contains "good" or try to extract reasoning
    decoded_lower = decoded.lower()
    if "good" in decoded_lower and len(decoded) < 20:
        return "good"
    
    # If no JSON found, return the decoded text (might be reasoning)
    if decoded:
        return decoded[:200]  # Limit length
    
    return ""


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
