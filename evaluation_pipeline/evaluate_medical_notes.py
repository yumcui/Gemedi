#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Notes Evaluation Script

This script evaluates medical notes from a CSV file using two methods:
1. Realism score evaluation (integrated realism logic)
2. Gemini API medical reasoning evaluation

The script adds three columns to the CSV:
- realism_score: Realism evaluation score (weighted 0.5)
- gemini_medical_reasoning_evaluation: Gemini API score (weighted 0.5)
- overall_score: Combined score (realism_score * 0.5 + gemini_score * 0.5)

Usage:
    python evaluate_medical_notes.py <input_csv_file> [output_csv_file]
    
Environment Variables:
    GEMINI_API_KEY: Required. Your Google Gemini API key.
"""

import sys
import os
import pandas as pd
import google.generativeai as genai
import json
import re
import time
import math
from datetime import datetime, date
from dateutil import parser
from enum import Enum
from typing import Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque

# Add preprocess.py directory to path to import realism evaluation functions
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_dir = os.path.join(script_dir, '..', 'generate_and_preprocess')
sys.path.insert(0, preprocess_dir)

# Import realism evaluation functions from preprocess.py
from preprocess import (
    calculate_document_structure_score,
    compute_linguistic_for_row,
    compute_difficulty_for_row,
    calculate_age_at_discharge,
    extract_sex,
    extract_diagnosis,
    ClinicalConsistencyEvaluator,
    ConsistencyCheck,
    LINGUISTIC_WEIGHT,
    CLINICAL_WEIGHT,
    STRUCTURE_WEIGHT,
    MODEL_NAME
)

# Check if ollama is available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ============================================================
# Extract PHI from text (needed for evaluation)
# ============================================================


def extract_phi_from_text(text: str) -> list:
    """Extract PHI information from text and build phi_annotations"""
    phi_annotations = []
    
    # Extract name
    name_match = re.search(r'Name:\s+([A-Za-z\s]+?)(?:\s+Unit No:|$)', text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        if name:
            phi_annotations.append({
                "value": name,
                "type": "NAME",
                "reason": "Patient's name extracted from text"
            })
    
    # Extract Unit No
    unit_match = re.search(r'Unit No:\s*(\d+)', text, re.IGNORECASE)
    if unit_match:
        unit_no = unit_match.group(1).strip()
        phi_annotations.append({
            "value": unit_no,
            "type": "UNIT_NO",
            "reason": "Unit number extracted from text"
        })
    
    # Extract Admission Date
    adm_match = re.search(r'Admission Date:\s*([0-9\-\/]+)', text, re.IGNORECASE)
    if adm_match:
        adm_date = adm_match.group(1).strip()
        phi_annotations.append({
            "value": adm_date,
            "type": "DATE_OF_ADMISSION",
            "reason": "Admission date extracted from text"
        })
    
    # Extract Discharge Date
    dis_match = re.search(r'Discharge Date:\s*([0-9\-\/]+)', text, re.IGNORECASE)
    if dis_match:
        dis_date = dis_match.group(1).strip()
        phi_annotations.append({
            "value": dis_date,
            "type": "DISCHARGE_DATE",
            "reason": "Discharge date extracted from text"
        })
    
    # Extract Date of Birth
    dob_match = re.search(r'(?:Date of Birth|DOB):\s*([0-9\-\/]+)', text, re.IGNORECASE)
    if dob_match:
        dob = dob_match.group(1).strip()
        phi_annotations.append({
            "value": dob,
            "type": "DATE_OF_BIRTH",
            "reason": "Date of birth extracted from text"
        })
    
    # Extract Sex
    sex_match = re.search(r'Sex:\s*([MF])', text, re.IGNORECASE)
    if sex_match:
        sex = sex_match.group(1).strip()
        phi_annotations.append({
            "value": sex,
            "type": "SEX",
            "reason": "Sex extracted from text"
        })
    
    # Extract Attending Doctor
    attending_match = re.search(r'Attending:\s*(Dr\.?\s+[A-Za-z\s]+)', text, re.IGNORECASE)
    if attending_match:
        doctor = attending_match.group(1).strip()
        phi_annotations.append({
            "value": doctor,
            "type": "PROVIDER_NAME",
            "reason": "Attending physician extracted from text"
        })
    
    return phi_annotations




# ============================================================
# Main Realism Evaluation Function
# ============================================================

def evaluate_realism(text: str) -> tuple:
    """
    Evaluate realism score for a given text.
    Returns (realism_score, details_dict) where details_dict contains error information.
    """
    # Extract PHI information
    phi_annotations = extract_phi_from_text(text)
    
    details = {
        "linguistic_issues": [],
        "document_structure_issues": [],
        "clinical_consistency_issues": []
    }
    
    # MODULE 1: Document Structure
    doc_score, doc_reason = calculate_document_structure_score(text)
    if doc_score < 1.0:
        # Extract issues from doc_reason
        if "format error" in doc_reason.lower():
            details["document_structure_issues"].append("Document format issues")
        if "missing" in doc_reason.lower():
            details["document_structure_issues"].append("Missing required keywords")
    
    # MODULE 2: Linguistic & Difficulty
    linguistic_result = compute_linguistic_for_row(phi_annotations)
    if isinstance(linguistic_result, tuple):
        linguistic, linguistic_details = linguistic_result
    else:
        linguistic = linguistic_result
        linguistic_details = []
    
    if linguistic_details and linguistic_details != ["No violations"]:
        details["linguistic_issues"] = linguistic_details
    
    phi_amount, ambiguity, difficulty = compute_difficulty_for_row(phi_annotations)
    
    # MODULE 3: Clinical Consistency (optional)
    age = calculate_age_at_discharge(text)
    sex = extract_sex(text)
    diagnosis = extract_diagnosis(text)
    
    clinical_consistency = 0.0
    clinical_issues = []
    
    if age != -1 and sex and diagnosis and OLLAMA_AVAILABLE:
        try:
            # Parallel execution of age and sex checks using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                age_future = executor.submit(
                    ClinicalConsistencyEvaluator.get_relevance,
                    ConsistencyCheck.AGE,
                    str(age),
                    diagnosis,
                    text
                )
                sex_future = executor.submit(
                    ClinicalConsistencyEvaluator.get_relevance,
                    ConsistencyCheck.SEX,
                    sex,
                    diagnosis,
                    text
                )
                
                # Get results from futures
                age_score = age_future.result()
                sex_score = sex_future.result()
            
            if age_score != -1 and sex_score != -1:
                age_score_norm = age_score / 10.0
                sex_score_norm = sex_score / 10.0
                
                if age_score_norm < 0.5:
                    clinical_issues.append(f"Age-diagnosis mismatch (age_score: {age_score_norm:.2f})")
                if sex_score_norm < 0.5:
                    clinical_issues.append(f"Sex-diagnosis mismatch (sex_score: {sex_score_norm:.2f})")
                
                # Calculate clinical_consistency_realism
                if age_score_norm < 0.3:
                    clinical_consistency = age_score_norm * 1.2
                elif age_score_norm < 0.5:
                    age_for_calc = max(0.01, age_score_norm)
                    sex_for_calc = max(0.01, sex_score_norm)
                    geometric_mean = math.sqrt(age_for_calc * sex_for_calc)
                    clinical_consistency = geometric_mean * 0.6
                else:
                    clinical_consistency = (age_score_norm * 0.6 + sex_score_norm * 0.4)
            else:
                clinical_issues.append("LLM returned -1")
        except Exception as e:
            clinical_issues.append(f"Clinical consistency error: {str(e)[:50]}")
    elif age == -1:
        clinical_issues.append("Failed to extract age")
    elif not sex:
        clinical_issues.append("Failed to extract sex")
    elif not diagnosis or diagnosis == "Unknown Diagnosis":
        clinical_issues.append("Failed to extract diagnosis")
    elif not OLLAMA_AVAILABLE:
        clinical_issues.append("Ollama not available")
    
    if clinical_issues:
        details["clinical_consistency_issues"] = clinical_issues
    
    # Note: clinical_consistency remains 0.0 if not available (matches test_text_quality.py exactly)
    # It will be handled by max(0.01, clinical_consistency) in the calculation below
    
    # Calculate final realism
    ling = max(0.01, linguistic)
    clin = max(0.01, clinical_consistency)
    struct = max(0.01, doc_score)
    
    # Weighted geometric mean
    log_realism = (
        LINGUISTIC_WEIGHT * math.log(ling) +
        CLINICAL_WEIGHT * math.log(clin) +
        STRUCTURE_WEIGHT * math.log(struct)
    )
    geometric_mean = math.exp(log_realism)
    
    # Mixed approach: 70% geometric mean + 30% arithmetic mean
    arithmetic_mean = (
        LINGUISTIC_WEIGHT * linguistic +
        CLINICAL_WEIGHT * clinical_consistency +
        STRUCTURE_WEIGHT * doc_score
    )
    
    realism = 0.7 * geometric_mean + 0.3 * arithmetic_mean
    realism_score = max(0.0, min(1.0, realism))
    
    # Format details as a string
    all_issues = []
    if details["linguistic_issues"]:
        all_issues.append(f"Linguistic: {'; '.join(details['linguistic_issues'][:3])}")
    if details["document_structure_issues"]:
        all_issues.append(f"Structure: {'; '.join(details['document_structure_issues'][:2])}")
    if details["clinical_consistency_issues"]:
        all_issues.append(f"Clinical: {'; '.join(details['clinical_consistency_issues'][:2])}")
    
    if not all_issues:
        details_str = "No issues found"
    else:
        details_str = " | ".join(all_issues)
    
    return (realism_score, details_str)


# ============================================================
# Gemini API Evaluation with Rate Limiting
# ============================================================

# Rate limiter for Gemini API (free tier: 10 requests per minute)
class RateLimiter:
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Rate limiter for API calls.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            # Remove requests older than time_window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # If we've hit the limit, wait until the oldest request expires
            if len(self.requests) >= self.max_requests:
                wait_time = self.time_window - (now - self.requests[0]) + 1
                if wait_time > 0:
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(time.time())

# Global rate limiter instance
gemini_rate_limiter = RateLimiter(max_requests=8, time_window=60)  # Use 8 to be safe


def get_gemini_score(note_text: str, api_key: str, max_retries: int = 3) -> float:
    """
    Evaluate medical note using Gemini API based on audit criteria.
    Returns a score from 0-1 based on the audit results.
    
    Args:
        note_text: Medical note text to evaluate
        api_key: Google Gemini API key
        max_retries: Maximum number of retries for rate limit errors
    """
    # Configure once per thread (thread-safe)
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass  # Already configured
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are a Medical Quality Assurance Auditor. Your task is to evaluate the overall medical reasoning quality of the following medical note.
    
    Evaluate the note based on THREE dimensions of medical reasoning:
    
    1. Clinical Logic: Assess how medically sensible and realistic the clinical scenario is.
       - Consider: Are the diagnoses appropriate for the patient's age, sex, and clinical presentation?
       - Consider: Are the disease combinations biologically plausible?
       - Higher score = More clinically sensible and realistic
       - Lower score = Biologically impossible or highly implausible combinations
    
    2. Internal Consistency: Assess how well different parts of the note align with each other.
       - Consider: Do discharge instructions match the diagnosis and condition?
       - Consider: Are treatments and medications consistent with the clinical picture?
       - Consider: Do dates and timelines make sense?
       - Higher score = All parts are well-aligned and consistent
       - Lower score = Contradictions or misalignments between different sections
    
    3. Medication Inconsistency: Assess the appropriateness of medications given the diagnosis.
       - Consider: Are standard medications for the diagnosis present?
       - Consider: Are medications appropriate for the condition?
       - Consider: Are there missing critical medications or inappropriate ones?
       - Higher score = Medications are appropriate and complete for the diagnosis
       - Lower score = Missing standard medications or inappropriate medications
    
    Input Medical Note:
    \"\"\"
    {note_text}
    \"\"\"

    Evaluation Approach:
    - Evaluate the OVERALL REASONING QUALITY, not just count errors
    - A note can have minor issues but still be generally sensible and coherent
    - Focus on how well the note makes sense as a whole, not on finding every small error
    - Consider the severity and impact of any issues, not just their presence
    
    Scoring Guidelines:
    - Score 0.9-1.0: Highly sensible and consistent medical reasoning. All three dimensions are well-aligned.
    - Score 0.7-0.8: Generally sensible with minor inconsistencies or minor gaps. Overall makes good medical sense.
    - Score 0.5-0.6: Moderately sensible but has some notable inconsistencies or gaps. Some parts don't align well.
    - Score 0.3-0.4: Has significant issues that affect medical reasoning quality. Multiple inconsistencies or implausibilities.
    - Score 0.0-0.2: Highly implausible or contradictory. Major biological impossibilities or severe inconsistencies.
    
    Output Format:
    - Provide ONLY a score from 0.0 to 1.0
    - Format: "Score: X.X" where X.X is your evaluation score
    - Optionally, provide brief reasoning: "Score: X.X | Brief reason"
    - Do not list individual errors unless they are critical to understanding the score
    
    Example Outputs:
    Score: 0.95 | All dimensions well-aligned, highly sensible clinical reasoning
    Score: 0.75 | Generally sensible with minor medication gaps
    Score: 0.55 | Some inconsistencies between diagnosis and instructions
    Score: 0.35 | Significant clinical logic issues and inconsistencies
    Score: 0.15 | Major biological impossibilities
    """
    
    for attempt in range(max_retries + 1):
        try:
            # Apply rate limiting
            gemini_rate_limiter.wait_if_needed()
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract score from response
            score_match = re.search(r'Score:\s*([0-9.]+)', response_text, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
                return score
            
            # If no score found, check for "No errors found"
            if "no errors found" in response_text.lower():
                return 1.0
            
            # If errors are mentioned but no score, estimate based on error count
            error_count = len([line for line in response_text.split('\n') if ':' in line and any(cat in line.lower() for cat in ['clinical', 'internal', 'medication'])])
            if error_count == 0:
                return 1.0
            elif error_count == 1:
                return 0.7
            elif error_count == 2:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                # Try to extract retry delay from error message
                retry_delay_match = re.search(r'retry.*?(\d+\.?\d*)\s*seconds?', error_str, re.IGNORECASE)
                if retry_delay_match:
                    wait_time = float(retry_delay_match.group(1))
                else:
                    # Default wait time based on attempt
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s...
                
                if attempt < max_retries:
                    print(f"\nWarning: Rate limit exceeded. Waiting {wait_time:.1f}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"\nWarning: Rate limit exceeded after {max_retries} retries. Returning default score.")
                    return 0.0
            else:
                # Other errors
                if attempt < max_retries:
                    print(f"\nWarning: Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                    continue
                else:
                    print(f"\nWarning: Gemini API error after {max_retries} retries: {e}")
                    return 0.0
    
    return 0.0


# ============================================================
# Main Evaluation Function
# ============================================================

def evaluate_single_row(args):
    """
    Evaluate a single row (realism + Gemini).
    Used for parallel processing.
    
    Args:
        args: Tuple of (index, text, api_key)
    
    Returns:
        Tuple of (index, realism_score, gemini_score, overall_score, realism_details)
    """
    index, text, api_key = args
    
    # Skip empty texts
    if pd.isna(text) or str(text).strip() == "":
        return (index, 0.0, 0.0, 0.0, "Empty text")
    
    text_str = str(text)
    
    # Get realism score and details
    realism_score, realism_details = evaluate_realism(text_str)
    
    # Get Gemini score
    gemini_score = get_gemini_score(text_str, api_key)
    
    # Calculate overall score (each weighted 0.5)
    overall_score = (realism_score * 0.5) + (gemini_score * 0.5)
    
    return (index, realism_score, gemini_score, overall_score, realism_details)


def evaluate_csv(input_csv: str, output_csv: str = None, max_workers: int = 2, num_rows: int = None):
    """
    Evaluate medical notes from CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional)
        max_workers: Maximum number of threads for parallel processing (default: 2)
        num_rows: Number of rows to process (optional, default: all rows)
    """
    # Check for Gemini API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Read CSV file
    print(f"Reading CSV file: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Auto-rename logic: if CSV has only one column, rename it to 'generated_text'
    if len(df.columns) == 1:
        single_column = df.columns[0]
        print(f"Note: CSV has only one column '{single_column}'. Renaming to 'generated_text'.")
        df.rename(columns={single_column: 'generated_text'}, inplace=True)
    
    # Check for generated_text column
    if 'generated_text' not in df.columns:
        # Try to find similar column names
        possible_names = ['generated_note', 'generated', 'text', 'note', 'content']
        found_column = None
        for col in df.columns:
            col_lower = col.lower()
            if 'generated' in col_lower or col_lower in possible_names:
                found_column = col
                break
        
        if found_column:
            print(f"Note: Column 'generated_text' not found, but found '{found_column}'. Using it.")
            df.rename(columns={found_column: 'generated_text'}, inplace=True)
        else:
            print(f"Error: Column 'generated_text' not found in CSV.")
            print(f"Available columns: {df.columns.tolist()}")
            print("Tip: If your CSV has only one column, it will be automatically renamed to 'generated_text'.")
            sys.exit(1)
    
    # Limit number of rows if specified
    original_row_count = len(df)
    if num_rows is not None and num_rows > 0:
        if num_rows < original_row_count:
            df = df.head(num_rows)
            print(f"Note: Processing only first {num_rows} rows (out of {original_row_count} total rows).")
        elif num_rows > original_row_count:
            print(f"Warning: Requested {num_rows} rows but CSV only has {original_row_count} rows. Processing all rows.")
    
    # Generate output filename if not provided
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_evaluated.csv"
    
    print(f"Evaluating {len(df)} medical notes...")
    if not OLLAMA_AVAILABLE:
        print("Note: ollama not available. Clinical consistency evaluation will use default values.")
    print(f"Using {max_workers} threads for parallel processing...")
    print("Note: Gemini API free tier limit is 10 requests/minute. Rate limiting is enabled.")
    
    # Prepare data for parallel processing
    # Use enumerate to get position index (0-based) instead of DataFrame index
    tasks = []
    for pos_idx, (_, row) in enumerate(df.iterrows()):
        text = row['generated_text']
        tasks.append((pos_idx, text, api_key))
    
    # Initialize result lists with None (to maintain order)
    realism_scores = [None] * len(df)
    gemini_scores = [None] * len(df)
    overall_scores = [None] * len(df)
    realism_details_list = [None] * len(df)
    
    # Process in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pos = {executor.submit(evaluate_single_row, task): task[0] for task in tasks}
        
        # Process completed tasks with progress bar
        completed = 0
        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for future in as_completed(future_to_pos):
                try:
                    pos_idx, realism_score, gemini_score, overall_score, realism_details = future.result()
                    realism_scores[pos_idx] = realism_score
                    gemini_scores[pos_idx] = gemini_score
                    overall_scores[pos_idx] = overall_score
                    realism_details_list[pos_idx] = realism_details
                    completed += 1
                    pbar.update(1)
                except Exception as e:
                    pos_idx = future_to_pos[future]
                    print(f"\nWarning: Error processing row {pos_idx}: {e}")
                    realism_scores[pos_idx] = 0.0
                    gemini_scores[pos_idx] = 0.0
                    overall_scores[pos_idx] = 0.0
                    realism_details_list[pos_idx] = f"Error: {str(e)[:100]}"
                    completed += 1
                    pbar.update(1)
    
    # Add columns to dataframe
    df['realism_score'] = realism_scores
    df['gemini_medical_reasoning_evaluation'] = gemini_scores
    df['overall_score'] = overall_scores
    df['realism_issues'] = realism_details_list
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Calculate averages
    avg_realism = sum(realism_scores) / len(realism_scores) if realism_scores else 0.0
    avg_gemini = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0.0
    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    
    # Build summary text
    summary_lines = []
    summary_lines.append("\n" + "="*60)
    summary_lines.append("Evaluation Summary")
    summary_lines.append("="*60)
    summary_lines.append(f"Input CSV: {input_csv}")
    summary_lines.append(f"Output CSV: {output_csv}")
    summary_lines.append(f"Total Rows Evaluated: {len(df)}")
    summary_lines.append("")
    summary_lines.append(f"Average Realism Score: {avg_realism:.4f}")
    summary_lines.append(f"Average Gemini Medical Reasoning Score: {avg_gemini:.4f}")
    summary_lines.append(f"Average Overall Score: {avg_overall:.4f}")
    summary_lines.append("="*60)
    
    # Print summary
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary to txt file
    summary_txt_file = os.path.splitext(output_csv)[0] + "_summary.txt"
    try:
        with open(summary_txt_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\nSummary saved to: {summary_txt_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save summary to txt file: {e}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python evaluate_medical_notes.py <input_csv_file> [output_csv_file] [max_workers] [num_rows]")
        print("\nExample:")
        print("  python evaluate_medical_notes.py generated_medical_notes.csv")
        print("  python evaluate_medical_notes.py generated_medical_notes.csv output.csv 2")
        print("  python evaluate_medical_notes.py generated_medical_notes.csv output.csv 2 50")
        print("  python evaluate_medical_notes.py generated_medical_notes.csv 2 25  # Skip output file, use 2 threads, 25 rows")
        print("\nArguments:")
        print("  input_csv_file: Path to input CSV file (required)")
        print("  output_csv_file: Path to output CSV file (optional, default: <input>_evaluated.csv)")
        print("  max_workers: Number of threads for parallel processing (optional, default: 2)")
        print("              Note: Gemini free tier allows 10 requests/minute. Use 1-2 threads to avoid rate limits.")
        print("  num_rows: Number of rows to process (optional, default: all rows)")
        print("           Useful for testing with a subset of data.")
        print("\nNote: Requires GEMINI_API_KEY environment variable to be set.")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = None
    max_workers = 2
    num_rows = None
    
    # Smart parameter parsing: detect if arguments are numbers or file paths
    # Arguments: [input_csv] [output_csv?] [max_workers?] [num_rows?]
    arg_idx = 2
    
    # Check if second argument is a number (max_workers) or a file path (output_csv)
    if len(sys.argv) > arg_idx:
        arg2 = sys.argv[arg_idx]
        # If it's a pure number, treat it as max_workers (skipping output_csv)
        if arg2.isdigit():
            max_workers = int(arg2)
            arg_idx += 1
        # Otherwise, treat it as output_csv
        else:
            output_csv = arg2
            arg_idx += 1
            # Next argument should be max_workers
            if len(sys.argv) > arg_idx:
                try:
                    max_workers = int(sys.argv[arg_idx])
                    arg_idx += 1
                except ValueError:
                    print(f"Warning: Invalid max_workers value '{sys.argv[arg_idx]}'. Using default: 2")
    
    # Parse num_rows (always the last numeric argument)
    if len(sys.argv) > arg_idx:
        try:
            num_rows = int(sys.argv[arg_idx])
            if num_rows <= 0:
                print(f"Warning: num_rows must be positive. Processing all rows.")
                num_rows = None
        except ValueError:
            print(f"Warning: Invalid num_rows value '{sys.argv[arg_idx]}'. Processing all rows.")
            num_rows = None
    
    evaluate_csv(input_csv, output_csv, max_workers, num_rows)


if __name__ == "__main__":
    main()
