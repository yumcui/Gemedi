#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call both fine-tuned discriminator and Medical Reasoning discriminator.

This script reads a CSV file with medical notes, calls both models,
and outputs a CSV with 3 columns:
1. medical_note: Original medical note
2. realism_reasoning: Output from fine-tuned llama3-discriminator (your LoRA)
3. medical_reasoning: Output from Medical Reasoning Model (Llama 3.1) as a detailed "reason" string
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
import google.generativeai as genai
import time
from collections import deque
from threading import Lock

# ------------------------------------------------------------------------
# Rate Limiter for Gemini
# ------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            if len(self.requests) >= self.max_requests:
                wait_time = self.time_window - (now - self.requests[0]) + 1
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.time_window:
                        self.requests.popleft()
            self.requests.append(time.time())

gemini_rate_limiter = RateLimiter(max_requests=8, time_window=60)

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"   # For fine-tuned model (realism)
# MEDICAL_REASONING_MODEL removed in favor of Gemini API
GEMINI_MODEL_NAME = "gemini-2.5-flash"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "llama3-discriminator")

# Default paths (can be overridden by command line arguments)
DEFAULT_INPUT_CSV = os.path.join(SCRIPT_DIR, "generated_medical_notes_cot.csv")
DEFAULT_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "both_discriminator_output.csv")

# ------------------------------------------------------------------------
# 1. PROMPTS
# ------------------------------------------------------------------------

# Fine-tuned discriminator prompt (realism reasoning) - 你自己的 LoRA 用的
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


# Medical Reasoning discriminator prompt (was BioMistral)
MEDICAL_REASONING_SYSTEM_PROMPT = """
You are an expert medical discriminator specializing in evaluating generated medical notes.

Your task is to analyze the provided medical note and identify all medical reasoning errors, factual inconsistencies, and clinical logic issues.

Please provide a detailed list of errors to help the user correct the note.
- If the note is logically correct → output {"reason": "good"}
- If there are errors → output {"reason": "1. Error 1... 2. Error 2..."} (detailed explanation)

STRICT RULES:
- Output MUST be valid JSON.
- Output must contain exactly one key: "reason".
- The value of "reason" should be a detailed string describing the errors.
- Do NOT output tags, labels, or multiple fields.
- Do NOT copy or restate the instructions or the note outside the JSON object.
- Do NOT add any explanation before or after the JSON.

Examples:
{"reason": "good"}
{"reason": "1. Timeline error: discharge date (2023-01-01) is before admission date (2023-01-05). 2. Medication mismatch: Patient has Penicillin allergy but Amoxicillin was prescribed."}
"""

# ------------------------------------------------------------------------
# 2. Load Models
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


def configure_gemini():
    """Configure Gemini API"""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully!")
    return api_key

# ------------------------------------------------------------------------
# 3. Inference Functions
# ------------------------------------------------------------------------

def generate_fine_tuned_prediction(model, tokenizer, note_content):
    """Generate prediction using fine-tuned discriminator (realism reasoning)"""
    combined_prompt = f"{FINE_TUNED_SYSTEM_PROMPT}\n\nMedical Note:\n---\n{note_content}\n---"
    
    # 使用 system+user（Llama 3.1 支持）
    try:
        messages = [
            {"role": "system", "content": FINE_TUNED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Medical Note:\n---\n{note_content}\n---"},
        ]
        
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        # fallback：只有 user，一次性拼 prompt
        messages = [
            {"role": "user", "content": combined_prompt},
        ]
        try:
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            input_text = combined_prompt
    
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


def generate_gemini_prediction(note_content, max_retries=3):
    """
    Use Gemini API for medical reasoning.
    Target: Output JSON string like {"reason": "..."}.
    """
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    prompt = f"""
{MEDICAL_REASONING_SYSTEM_PROMPT}

Medical Note:
---
{note_content}
---

Return ONLY the JSON object.
"""

    for attempt in range(max_retries + 1):
        try:
            # Apply rate limiting
            gemini_rate_limiter.wait_if_needed()
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                wait_time = (attempt + 1) * 10
                if attempt < max_retries:
                    print(f"\nWarning: Gemini rate limit exceeded. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # Other errors
            if attempt < max_retries:
                print(f"\nWarning: Gemini API error (attempt {attempt + 1}): {e}")
                time.sleep(2)
                continue
            
            return f'{{"reason": "ERROR: Gemini API failed: {str(e)[:100]}"}}'

    return '{"reason": "ERROR: Gemini API failed after retries"}'

# ------------------------------------------------------------------------
# 4. Parse Responses
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


def parse_base_model_response(text):
    """Extract {"reason": "..."} from model output."""
    text = text.strip()

    # Try to find a JSON-like object containing "reason"
    json_match = re.search(r'\{[\s\S]*?"reason"[\s\S]*?\}', text, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(0).strip()
        try:
            data = json.loads(json_text)
            reason = data.get("reason", "").strip()
            return reason
        except json.JSONDecodeError:
            # Try to extract "reason" value directly if JSON parse fails
            reason_match = re.search(r'"reason"\s*:\s*"([\s\S]*?)"', json_text)
            if reason_match:
                return reason_match.group(1)
            return f"ERROR: JSON parse failed ({json_text[:100]})"
    else:
        # Fallback
        if "good" in text.lower() and len(text) < 50:
            return "good"
        return f"ERROR: could not parse ({text[:200]})"

# ------------------------------------------------------------------------
# 5. Main Function
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
    print("Both Discriminator Evaluation (LoRA Llama + Gemini API)")
    print("="*60)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print("="*60)
    
    # Load models and configure API
    print("\nLoading fine-tuned model and configuring Gemini...")
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
    configure_gemini()
    
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
                if isinstance(note_content, str) and note_content.startswith('"') and note_content.endswith('"'):
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
            if not note_content or not str(note_content).strip():
                writer.writerow([note_content, "", ""])
                f_out.flush()
                continue
            
            try:
                # Call fine-tuned discriminator (realism reasoning)
                try:
                    fine_tuned_raw = generate_fine_tuned_prediction(
                        fine_tuned_model, fine_tuned_tokenizer, note_content
                    )
                    realism_reasoning = parse_fine_tuned_response(fine_tuned_raw)
                except Exception as e1:
                    print(f"\nWarning: Fine-tuned model error for note {i+1}: {e1}")
                    realism_reasoning = f"ERROR: {str(e1)[:100]}"
                
                # Call Gemini API (medical reasoning)
                try:
                    base_raw = generate_gemini_prediction(note_content)
                    medical_reasoning = parse_base_model_response(base_raw)
                except Exception as e2:
                    print(f"\nWarning: Gemini API error for note {i+1}: {e2}")
                    medical_reasoning = f"ERROR: {str(e2)[:100]}"
                
                # Write to CSV
                writer.writerow([note_content, realism_reasoning, medical_reasoning])
                f_out.flush()
                
            except Exception as e:
                print(f"\nError processing note {i+1}: {e}")
                import traceback
                traceback.print_exc()
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
    print("  2. realism_reasoning: Output from fine-tuned llama3-discriminator (LoRA)")
    print("  3. medical_reasoning: Output from Gemini API (detailed feedback)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
