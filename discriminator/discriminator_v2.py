import csv
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "BioMistral/BioMistral-7B"
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
# 2. PROMPT TEMPLATE (升级版，结构化 JSON 输出)
# ================================================================
SYSTEM_PROMPT = """
You are an expert clinical documentation auditor specializing in detecting medical logic errors and inconsistencies in discharge summaries and clinical notes.

Your task is to analyze the medical note and identify any of the following types of errors:

1. Clinical Logic Errors:
   - Age/sex mismatch with diagnosis (e.g., pediatric condition in an elderly patient, male-only condition in a female).
   - Biologically implausible disease combinations.
   - Clearly inappropriate primary diagnosis for the patient demographics.

2. Medication Logic Errors:
   - Prescribed medications that clearly do not match the main diagnoses.
   - Missing standard first-line medications for a major diagnosis when they are strongly expected.
   - Clearly inappropriate medication for patient age/organ function (e.g., contraindicated or unsafe).
   - Obvious dosage/route/frequency inconsistencies (e.g., impossible dose, unsafe frequency).

3. Timeline Errors:
   - Admission date is the same as or after the discharge date.
   - Dates that are impossible or inconsistent (e.g., DOB after admission date).
   - Impossible temporal sequences inside the note.

4. PHI (Protected Health Information) Inconsistencies:
   - DOB vs admission/discharge dates leading to an implausible age.
   - Date format inconsistencies that change the meaning (e.g., swapping month/day vs day/month).
   - Clearly invalid patient name formats (e.g., empty, single letter, or non-alphabetic garbage).

5. Internal Consistency Errors:
   - Contradictions between sections (e.g., diagnosis says “no surgery” but procedure section documents surgery).
   - Discharge instructions that clearly do not match the documented diagnoses or hospital course.
   - Treatment plan inconsistencies (e.g., instructing the patient to stop a drug that was never started).

Important principles:
- Treat every note as describing a REAL patient, even if the wording looks templated or synthetic.
- Focus ONLY on factual clinical or logical errors, not on style or wording.
- For pediatric patients:
  - Do NOT mark a medication as an error just because it is more common in adults (e.g., metformin in obese adolescents).
  - Only flag pediatric medication use when it is clearly unsafe, contraindicated, or impossible for the described age.
- If you are NOT reasonably confident that something is a true error based on widely accepted medical practice, you MUST treat it as correct.

STRICT RULES:
- If you find NO errors, you must output:
  {
    "label": "good",
    "tags": [],
    "severity": 0.0,
    "explanation": "good"
  }

- If you find errors, you must output a single JSON object with:
  - "label": "error"
  - "tags": a non-empty array of short error type tags, such as:
    ["timeline_error", "medication_diagnosis_mismatch", "age_sex_diagnosis_mismatch",
     "medication_age_contraindicated", "phi_inconsistency", "internal_contradiction"]
  - "severity": a number between 0.0 and 1.0 (higher = more serious/unsafe).
  - "explanation": a concise explanation (<= 50 words) describing the specific issues.

- You MUST return only a single JSON object and nothing else.
  - No extra text.
  - No code fences.
  - No prefixes like "Here is the JSON".

OUTPUT FORMAT (EXACT SHAPE):

{
  "label": "good" OR "error",
  "tags": [/* zero or more short tags, empty only when label is "good" */],
  "severity": 0.0 to 1.0,
  "explanation": "good" OR "concise description of the medical logic error(s)"
}

Examples:
- Good note:
  {
    "label": "good",
    "tags": [],
    "severity": 0.0,
    "explanation": "good"
  }

- Timeline error:
  {
    "label": "error",
    "tags": ["timeline_error"],
    "severity": 0.9,
    "explanation": "Admission date (2024-09-15) is after discharge date (2024-09-10)."
  }

- Medication and age mismatch:
  {
    "label": "error",
    "tags": ["medication_age_contraindicated"],
    "severity": 0.8,
    "explanation": "Warfarin dose and INR target are unsafe for a 3-year-old child."
  }
"""


# ================================================================
# 3. INFERENCE
# ================================================================
def _default_error_result(reason: str):
    """Fallback result when parsing/generation fails."""
    return {
        "label": "error",
        "tags": ["parse_error"],
        "severity": 0.5,
        "explanation": reason[:200],
    }


def run_llama(model, tokenizer, note):
    """
    Run inference using chat template format (better for Llama 3.1 Instruct).
    Return a dict: {label, tags, severity, explanation}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Medical Note:\n---\n{note}\n---\n\nAnalyze this medical note for logic errors and inconsistencies."
        },
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Try to extract JSON object (first {...})
    json_match = re.search(r'\{[\s\S]*\}', decoded, re.DOTALL)
    if not json_match:
        # Fallback: very short "good" style answer
        dl = decoded.lower()
        if "good" in dl and len(decoded) < 30:
            return {
                "label": "good",
                "tags": [],
                "severity": 0.0,
                "explanation": "good",
            }
        return _default_error_result(f"Could not find JSON in: {decoded[:200]}")

    json_text = json_match.group(0).strip()

    # Try more robust JSON parsing (truncate if Extra data)
    while True:
        try:
            data = json.loads(json_text)
            break
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                last_brace_index = json_text.rfind("}")
                if last_brace_index != -1:
                    json_text = json_text[: last_brace_index + 1].strip()
                    continue
            return _default_error_result(f"JSONDecodeError: {e} | text: {json_text[:200]}")
        except Exception as e:
            return _default_error_result(f"JSON parse failed: {e} | text: {json_text[:200]}")

    # Normalize fields
    label = data.get("label", "error")
    if label not in ["good", "error"]:
        label = "error"

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]

    # severity: clip to [0.0, 1.0]
    severity = data.get("severity", 0.5)
    try:
        severity = float(severity)
    except Exception:
        severity = 0.5
    severity = max(0.0, min(1.0, severity))

    explanation = data.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = str(explanation)

    # If label is good, normalize to a canonical form
    if label == "good":
        tags = []
        severity = 0.0
        explanation = "good"

    return {
        "label": label,
        "tags": tags,
        "severity": severity,
        "explanation": explanation[:200],
    }


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
# 5. SAVE CSV (多几列：label/tags/severity/explanation)
# ================================================================
def save_to_csv(records, csv_path):
    print(f"\nSaving CSV → {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_note", "label", "tags", "severity", "explanation"])

        for rec in records:
            writer.writerow([
                rec["original"],
                rec["label"],
                ";".join(rec["tags"]),
                rec["severity"],
                rec["explanation"],
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
        result = run_llama(model, tokenizer, note)

        output_rows.append({
            "original": note.strip(),
            "label": result["label"],
            "tags": result["tags"],
            "severity": result["severity"],
            "explanation": result["explanation"],
        })

    save_to_csv(output_rows, OUTPUT_CSV)

    print("\nAll done! Output CSV:", OUTPUT_CSV)
