#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call both fine-tuned discriminator and base model discriminator_v2.

This script reads a CSV file with medical notes, calls both models,
and outputs a CSV with 3 columns:
1. medical_note: Original medical note
2. realism_reasoning: Output from fine-tuned llama3-discriminator
3. medical_reasoning: Output from base model (discriminator_v2 logic)
"""

import csv
import json
import torch
import re
import sys
import os
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "llama3-discriminator")

# Default paths (can be overridden by command line arguments)
DEFAULT_INPUT_CSV = os.path.join(SCRIPT_DIR, "generated_medical_notes_cot.csv")
DEFAULT_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "both_discriminator_output.csv")

# Fine-tuned discriminator prompt (realism reasoning)
FINE_TUNED_SYSTEM_PROMPT = """You are an expert clinical documentation auditor specializing in detecting:
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

# Base model discriminator_v2 prompt (medical reasoning)
BASE_MODEL_SYSTEM_PROMPT = """You are an expert clinical documentation auditor specializing in detecting medical logic errors and inconsistencies.

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

# ------------------------------------------------------------------------
# Load Models
# ------------------------------------------------------------------------
def load_fine_tuned_model():
    """Load the fine-tuned discriminator model (llama3-discriminator)"""
    print(f"Loading fine-tuned model: {BASE_MODEL_ID} + {ADAPTER_PATH} ...")
    
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

    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: Adapter path not found: {ADAPTER_PATH}")
        sys.exit(1)
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Fine-tuned model loaded successfully!")
    return model, tokenizer

def load_base_model():
    """Load the base model for discriminator_v2 logic"""
    print(f"Loading base model: {BASE_MODEL_ID} ...")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Base model loaded successfully!")
    return model, tokenizer

# ------------------------------------------------------------------------
# Inference Functions
# ------------------------------------------------------------------------
def generate_fine_tuned_prediction(model, tokenizer, note_content):
    """Generate prediction using fine-tuned discriminator (realism reasoning)"""
    messages = [
        {"role": "system", "content": FINE_TUNED_SYSTEM_PROMPT},
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

def generate_base_model_prediction(model, tokenizer, note_content):
    """Generate prediction using base model (discriminator_v2 logic, medical reasoning)"""
    messages = [
        {"role": "system", "content": BASE_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Medical Note:\n---\n{note_content}\n---\n\nAnalyze this medical note for logic errors and inconsistencies."}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return decoded

# ------------------------------------------------------------------------
# Parse Responses
# ------------------------------------------------------------------------
def parse_fine_tuned_response(response_text):
    """Parse fine-tuned model response (expects {"reason": "..."})"""
    response_text = response_text.strip()
    
    json_match = re.search(r'\{[\s\S]*?"reason"[\s\S]*?\}', response_text, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(0).strip()
        try:
            data = json.loads(json_text)
            reason = data.get("reason", response_text)
            return reason
        except json.JSONDecodeError:
            # Try to extract "reason" value directly
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', json_text)
            if reason_match:
                return reason_match.group(1)
            return response_text
    else:
        # No JSON found, return raw response
        return response_text

def parse_base_model_response(response_text):
    """Parse base model response (expects {"issues": "..."})"""
    response_text = response_text.strip()
    
    json_match = re.search(r'\{[\s\S]*?"issues"[\s\S]*?\}', response_text, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(0).strip()
        
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
                
                # Try to extract "issues" value directly
                issues_match = re.search(r'"issues"\s*:\s*"([^"]+)"', json_text)
                if issues_match:
                    return issues_match.group(1)
                break
            except Exception:
                break
    
    # Fallback: check if response contains "good"
    response_lower = response_text.lower()
    if "good" in response_lower and len(response_text) < 20:
        return "good"
    
    # Return decoded text (might be reasoning)
    if response_text:
        return response_text[:200]  # Limit length
    
    return ""

# ------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------
def main():
    """Main execution function"""
    # Parse command line arguments
    input_csv = DEFAULT_INPUT_CSV
    output_csv = DEFAULT_OUTPUT_CSV
    
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    print("="*60)
    print("Both Discriminator Evaluation")
    print("="*60)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print("="*60)
    
    # Load both models
    print("\nLoading models...")
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
    base_model, base_tokenizer = load_base_model()
    
    # Read CSV file
    print(f"\nReading CSV file: {input_csv}")
    notes = []
    column_name = None
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            column_name = reader.fieldnames[0] if reader.fieldnames else None
            
            for row in reader:
                note_content = row[column_name] if column_name else row[list(row.keys())[0]]
                # Handle CSV quoting
                if note_content.startswith('"') and note_content.endswith('"'):
                    note_content = note_content[1:-1]
                notes.append(note_content)
        
        print(f"Loaded {len(notes)} notes from CSV (column: '{column_name}')")
    except FileNotFoundError:
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Process notes with both models
    print(f"\nProcessing notes with both discriminator models...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["medical_note", "realism_reasoning", "medical_reasoning"])
        
        for i, note_content in enumerate(tqdm(notes, desc="Processing")):
            if not note_content or not note_content.strip():
                writer.writerow([note_content, "", ""])
                f_out.flush()
                continue
            
            try:
                # Call fine-tuned discriminator (realism reasoning)
                fine_tuned_raw = generate_fine_tuned_prediction(
                    fine_tuned_model, fine_tuned_tokenizer, note_content
                )
                realism_reasoning = parse_fine_tuned_response(fine_tuned_raw)
                
                # Call base model discriminator_v2 (medical reasoning)
                base_raw = generate_base_model_prediction(
                    base_model, base_tokenizer, note_content
                )
                medical_reasoning = parse_base_model_response(base_raw)
                
                # Write to CSV
                writer.writerow([note_content, realism_reasoning, medical_reasoning])
                f_out.flush()
                
            except Exception as e:
                print(f"\nError processing note {i+1}: {e}")
                writer.writerow([
                    note_content, 
                    f"ERROR: {str(e)[:100]}", 
                    f"ERROR: {str(e)[:100]}"
                ])
                f_out.flush()
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Total notes processed: {len(notes)}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    print("\nOutput columns:")
    print("  1. medical_note: Original medical note")
    print("  2. realism_reasoning: Output from fine-tuned llama3-discriminator")
    print("  3. medical_reasoning: Output from base model (discriminator_v2 logic)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
