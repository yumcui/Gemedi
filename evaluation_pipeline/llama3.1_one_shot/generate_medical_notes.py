#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Medical Notes using Llama 3.1 with One-Shot Learning

This script uses Ollama to call Llama 3.1 model and generate 200 medical notes
based on a one-shot example from sample_text.txt.

Usage:
    python generate_medical_notes.py [output_csv_file] [num_notes]
    
Example:
    python generate_medical_notes.py generated_notes.csv 200
"""

import sys
import os
import pandas as pd
import ollama
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Configuration
MODEL_NAME = "llama3.1"
DEFAULT_NUM_NOTES = 200
DEFAULT_OUTPUT_FILE = "generated_medical_notes.csv"

# One-shot example (from sample_text.txt)
ONE_SHOT_EXAMPLE = """Name: Julianne Slade                 Unit No:   2648644
 
Admission Date: 2025-01-31              Discharge Date:   2025-02-11
 
Date of Birth: 1971-03-23             Sex:   F
 
Service: ORTHOPEDICS
 
Allergies: 
No Known Allergies / Adverse Drug Reactions
 
Attending: Dr. Julianne Fletcher

Discharge Medications:
1. Acetaminophen 325 mg Tablet Sig: One (1) Tablet PO Q6H as needed for pain.
Disp:*60 Tablet(s)* Refills:*0*
2. Atenolol 50 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).
Disp:*30 Tablet(s)* Refills:*0*
3. Lisinopril 10 mg Tablet Sig: Two (2) Tablet PO DAILY 
(Daily).  
4. Omeprazole 20 mg Capsule, Delayed Release(E.C.) Sig: One (1) 
Capsule, Delayed Release(E.C.) PO DAILY (Daily).
Disp:*30 Capsule, Delayed Release(E.C.)(s)* Refills:*0*
5. Oxycodone 5 mg Tablet Sig: Two (2) Tablet PO Q4H as needed for pain.
Disp:*60 Tablet(s)* Refills:*0*

 
Discharge Disposition:
Home
 
Discharge Diagnosis:
Right hip osteoarthritis, Hypertension, Gastroesophageal reflux disease (GERD)

 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.

 
Discharge Instructions:
Please call the doctor if you have a temperature greater than 101.5, 
increased pain in the hip, redness, swelling, or inability to bear weight.

You may resume all regular home medications unless otherwise 
instructed.  

Take all new medications as prescribed by your physicians and do not 
stop any medication without consulting your physician.
 

Followup Instructions:
Please return to see Dr. Julianne Fletcher in two weeks for a routine check-up to assess progress and adjust treatment 
plan as necessary for optimal management of right hip osteoarthritis symptoms. If you experience increased pain or 
difficulty walking before then, contact our office to discuss potential earlier follow-up options. We look forward to 
supporting your continued recovery!"""


def generate_single_note(index: int) -> tuple:
    """
    Generate a single medical note using Llama 3.1 with one-shot learning.
    
    Args:
        index: Index of the note (for tracking)
    
    Returns:
        Tuple of (index, generated_text, success)
    """
    prompt = f"""You are a medical documentation assistant. Generate a realistic medical discharge summary following the exact format and structure of the example below.

Example:
{ONE_SHOT_EXAMPLE}

Now generate a NEW medical discharge summary with:
- Different patient name, unit number, dates, and demographics
- Different medical conditions and diagnoses
- Different medications and dosages
- Different discharge instructions and follow-up plans
- Maintain the same format, structure, and level of detail as the example
- Ensure all dates are realistic (admission date < discharge date, DOB < admission date)
- Ensure clinical consistency (age-appropriate conditions, sex-appropriate diagnoses)
- Use realistic medical terminology and professional language

Generate ONLY the medical note, no additional text or explanations:"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical documentation assistant. Generate realistic medical discharge summaries following the provided format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.7,  # Moderate creativity
                "num_predict": 2000,  # Allow longer outputs
            }
        )
        
        generated_text = response["message"]["content"].strip()
        
        # Clean up the response (remove any markdown formatting or extra text)
        if generated_text.startswith("```"):
            # Remove markdown code blocks if present
            lines = generated_text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            generated_text = '\n'.join(lines).strip()
        
        return (index, generated_text, True)
        
    except Exception as e:
        print(f"\nWarning: Error generating note {index}: {e}")
        return (index, "", False)


def generate_medical_notes(output_file: str = None, num_notes: int = DEFAULT_NUM_NOTES, max_workers: int = 5):
    """
    Generate medical notes using Llama 3.1 with one-shot learning.
    
    Args:
        output_file: Path to output CSV file
        num_notes: Number of notes to generate
        max_workers: Number of threads for parallel processing
    """
    # Set default output file
    if output_file is None:
        output_file = DEFAULT_OUTPUT_FILE
    
    print(f"Generating {num_notes} medical notes using Llama 3.1...")
    print(f"Using {max_workers} threads for parallel processing...")
    print(f"One-shot example loaded from sample_text.txt format\n")
    
    # Test ollama connection
    try:
        print("Testing Ollama connection...")
        test_response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            options={"num_predict": 10}
        )
        print(f"✓ Ollama connection successful (model: {MODEL_NAME})\n")
    except Exception as e:
        print(f"Error: Failed to connect to Ollama: {e}")
        print("Please ensure Ollama is running and llama3.1 model is installed.")
        print("Install with: ollama pull llama3.1")
        sys.exit(1)
    
    # Prepare tasks
    tasks = list(range(num_notes))
    
    # Initialize result list
    results = [None] * num_notes
    
    # Generate notes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(generate_single_note, idx): idx for idx in tasks}
        
        # Process completed tasks with progress bar
        successful = 0
        failed = 0
        with tqdm(total=num_notes, desc="Generating notes") as pbar:
            for future in as_completed(future_to_index):
                try:
                    index, generated_text, success = future.result()
                    if success and generated_text:
                        results[index] = generated_text
                        successful += 1
                    else:
                        results[index] = ""
                        failed += 1
                    pbar.update(1)
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"\nWarning: Error processing note {idx}: {e}")
                    results[idx] = ""
                    failed += 1
                    pbar.update(1)
                
                # Small delay to avoid overwhelming the model
                time.sleep(0.1)
    
    # Filter out empty results
    valid_results = [(i, text) for i, text in enumerate(results) if text and text.strip()]
    
    print(f"\nGeneration complete!")
    print(f"  Successful: {successful}/{num_notes}")
    print(f"  Failed: {failed}/{num_notes}")
    
    if not valid_results:
        print("Error: No valid notes were generated!")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'index': [i for i, _ in valid_results],
        'generated_text': [text for _, text in valid_results]
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total valid notes: {len(valid_results)}")
    
    # Print sample
    if valid_results:
        print("\n" + "="*60)
        print("Sample Generated Note (first 500 characters):")
        print("="*60)
        sample_text = valid_results[0][1][:500]
        print(sample_text)
        if len(valid_results[0][1]) > 500:
            print("...")
        print("="*60)


def main():
    """Main function"""
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = DEFAULT_OUTPUT_FILE
    
    if len(sys.argv) > 2:
        try:
            num_notes = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid number of notes: {sys.argv[2]}")
            print("Usage: python generate_medical_notes.py [output_csv_file] [num_notes] [max_workers]")
            sys.exit(1)
    else:
        num_notes = DEFAULT_NUM_NOTES
    
    if len(sys.argv) > 3:
        try:
            max_workers = int(sys.argv[3])
        except ValueError:
            print(f"Error: Invalid number of workers: {sys.argv[3]}")
            sys.exit(1)
    else:
        max_workers = 5
    
    generate_medical_notes(output_file, num_notes, max_workers)


if __name__ == "__main__":
    main()

