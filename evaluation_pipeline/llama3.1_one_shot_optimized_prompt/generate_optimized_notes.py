#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate medical notes using Llama 3.1 with optimized prompt and one-shot example.

This script combines:
1. Service-specific biological constraints (SERVICE_RULES)
2. Chain-of-thought reasoning prompts
3. One-shot learning with a high-quality example
"""

import sys
import os
import random
import datetime
import csv
import ollama
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 1. Service Rules ====================
SERVICE_RULES = {
    "OBSTETRICS": {
        "sex_weights": {"Female": 1.0, "Male": 0.0},
        "age_min": 18, "age_max": 45
    },
    "PEDIATRICS": {
        "sex_weights": {"Female": 0.5, "Male": 0.5},
        "age_min": 0, "age_max": 17
    },
    "UROLOGY": {
        "sex_weights": {"Male": 0.8, "Female": 0.2},
        "age_min": 40, "age_max": 85
    },
    "CARDIOLOGY": {
        "sex_weights": {"Male": 0.5, "Female": 0.5},
        "age_min": 45, "age_max": 90
    },
    "ORTHOPEDICS": {
        "sex_weights": {"Male": 0.5, "Female": 0.5},
        "age_min": 18, "age_max": 85
    },
    "NEUROLOGY": {
        "sex_weights": {"Male": 0.5, "Female": 0.5},
        "age_min": 25, "age_max": 90
    },
    "GENERAL_MEDICINE": {
        "sex_weights": {"Male": 0.5, "Female": 0.5},
        "age_min": 20, "age_max": 85
    }
}

# ==================== 2. Helper Functions ====================
def get_random_dob(age: int) -> str:
    """Generate a random date of birth for a given age"""
    today = datetime.date.today()
    birth_year = today.year - age
    start_date = datetime.date(birth_year, 1, 1)
    end_date = datetime.date(birth_year, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    dob = start_date + datetime.timedelta(days=random_days)
    return dob.strftime("%Y-%m-%d")

def load_one_shot_example() -> str:
    """Load the one-shot example from sample_text.txt"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_text_path = os.path.join(script_dir, '..', 'test', 'sample_text.txt')
    
    try:
        with open(sample_text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Sample text file not found at {sample_text_path}")
        print("Using a default example...")
        return """Name:  Ava Moreno                  Unit No:   5314445
 
Admission Date:  2024-06-14              Discharge Date:   2024-07-03
 
Date of Birth:  1991-07-06             Sex:   F
 
Service: NEUROLOGY
 
Allergies: 
No Known Allergies / Adverse Drug Reactions
 
Attending: Dr. Julianne Patel

Discharge Medications:
1. Acetaminophen 650 mg PO Q6H 
2. Docusate Sodium 100 mg PO BID 
3. Ondansetron 8 mg PO Q8H:PRN nausea 
4. Senna 17.2 mg PO BID 
5. Simvastatin 20 mg PO DAILY 
6. Aspirin 81 mg PO DAILY 
7. Metoprolol Succinate XL 25 mg PO DAILY 
8. Lisinopril 10 mg PO DAILY 
9. Fluticasone Propionate NASAL 1 SPRY NU BID:PRN congestion 

 
Discharge Disposition:
Home
 
Discharge Diagnosis:
Primary diagnosis:  Acute on Chronic Subarachnoid Hemorrhage
Secondary diagnoses:  Hypertension, Hyperlipidemia

 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.

 
Discharge Instructions:
Dear Ms. Moreno,

You were admitted to the hospital after a CT scan showed blood in your brain.
The neurosurgeons felt that you did not need surgery at this time, but
you will be followed by the neurologists for any changes or worsening of
your symptoms.

Please follow up with your primary care doctor as an outpatient and 
continue all of your home medications unless specifically instructed 
to stop by your doctor. 

Please call your doctor or return to the emergency department if you 
experience any of the following:
- New onset of tremors, weakness, numbness, tingling, seizures, 
difficulty with speech or movement.
- Any other worrisome symptoms that you feel your doctor should know 
about.

Thank you for allowing us to participate in your care,
Your Edwards Neurology Team
 
Followup Instructions:
Schedule a follow-up appointment with Dr. Julianne Patel in 1 week to monitor progress and adjust treatment plan as needed, or sooner if symptoms worsen or new concerns arise."""

# ==================== 3. Enhanced Prompt Construction ====================
def construct_detailed_prompt(sex: str, service: str, age: int, dob: str, one_shot_example: str) -> str:
    """
    Construct prompt with CoT, medication constraints, and one-shot example
    """
    
    # Determine if pediatric case
    is_pediatric = "YES" if age < 18 else "NO"
    
    prompt = f"""
### ROLE:
You are an expert Clinical Pharmacist and Medical Scribe. You prize accuracy, standard of care, and specific dosing.

### PATIENT PROFILE:
- **Service**: {service}
- **Sex**: {sex}
- **Age**: {age} (Pediatric Case: {is_pediatric})
- **DOB**: {dob}

### INSTRUCTION (CHAIN OF THOUGHT):
Before generating the note, perform the following reasoning steps implicitly:

1. **Diagnosis Selection**: Choose a common, realistic diagnosis for this specific Age/Sex/Service combination.

2. **Treatment Plan**: Identify the Gold Standard treatment.

3. **Dosage Calculation**: 
   - If Pediatric: Calculate dosage based on weight (simulate a reasonable weight). Ensure units are correct (e.g., mg/kg).
   - If Adult: Use standard fixed dosing (e.g., 500mg, 1000mg).

4. **Safety Check**: Verify the medication route and frequency are logical for a discharge summary (home meds, not IV drips unless home infusion).

### STRICT MEDICATION CONSTRAINTS:
1. **No Ranges**: Do not write "1-2 tablets". Write exactly "1 tablet" or "2 tablets".

2. **Complete Sig**: You MUST include: Drug Name + Strength + Route + Frequency.
   - *Bad*: "Amoxicillin as needed"
   - *Good*: "Amoxicillin 500 mg PO (by mouth) TID (3 times a day) for 7 days."

3. **Consistency**: Ensure the drug matches the diagnosis. (e.g., Don't prescribe antibiotics for a broken leg unless post-op infection risk).

### REQUIRED OUTPUT FORMAT:
Generate a Hospital Discharge Summary with these headers:
1. **Patient Demographics** (Name, DOB, Sex, etc.)
2. **Admission Diagnosis**
3. **Discharge Diagnosis** (Must be specific, ICD-10 style)
4. **Hospital Course** (Concise summary of stay)
5. **Discharge Medications** (List as bullet points, STRICT FORMATTING required)
6. **Discharge Instructions**

### EXAMPLE (One-Shot Learning):
Here is a high-quality example of the format and style you should follow:

\"\"\"
{one_shot_example}
\"\"\"

### TASK:
Generate a NEW medical discharge summary following the same format and quality as the example above, but for a patient with:
- Service: {service}
- Sex: {sex}
- Age: {age}
- DOB: {dob}

Ensure the diagnosis, medications, and instructions are appropriate for this specific patient profile.

### OUTPUT:
Generate the medical note now.
"""
    return prompt

# ==================== 4. Generation Function ====================
def generate_single_note(args):
    """Generate a single medical note"""
    index, sex, service, age, dob, one_shot_example, model_name = args
    
    try:
        prompt = construct_detailed_prompt(sex, service, age, dob, one_shot_example)
        
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 2000
            }
        )
        
        generated_text = response['response'].strip()
        return (index, generated_text, None)
    except Exception as e:
        return (index, "", str(e))

def generate_biologically_valid_prompts(num_records: int) -> List[tuple]:
    """Generate patient profiles based on SERVICE_RULES"""
    profiles = []
    service_keys = list(SERVICE_RULES.keys())
    
    for i in range(num_records):
        service = random.choice(service_keys)
        rule = SERVICE_RULES[service]
        
        # Select sex based on weights
        sex = random.choices(
            list(rule["sex_weights"].keys()), 
            weights=list(rule["sex_weights"].values())
        )[0]
        
        # Generate age within service range
        age = random.randint(rule["age_min"], rule["age_max"])
        dob = get_random_dob(age)
        
        profiles.append((i, sex, service, age, dob))
    
    return profiles

# ==================== 5. Main Function ====================
def main():
    """Main function"""
    # Parse arguments
    output_csv = "generated_medical_notes.csv"
    num_notes = 50
    max_workers = 5
    model_name = "llama3.1"
    
    if len(sys.argv) > 1:
        output_csv = sys.argv[1]
    if len(sys.argv) > 2:
        num_notes = int(sys.argv[2])
    if len(sys.argv) > 3:
        max_workers = int(sys.argv[3])
    
    print("="*60)
    print("Llama 3.1 Optimized Prompt Medical Notes Generation")
    print("="*60)
    print(f"Output file: {output_csv}")
    print(f"Number of notes: {num_notes}")
    print(f"Model: {model_name}")
    print(f"Max workers: {max_workers}")
    print("="*60)
    
    # Load one-shot example
    print("\nLoading one-shot example...")
    one_shot_example = load_one_shot_example()
    print("One-shot example loaded successfully.")
    
    # Generate patient profiles
    print(f"\nGenerating {num_notes} patient profiles...")
    profiles = generate_biologically_valid_prompts(num_notes)
    
    # Prepare tasks
    tasks = []
    for index, sex, service, age, dob in profiles:
        tasks.append((index, sex, service, age, dob, one_shot_example, model_name))
    
    # Generate notes in parallel
    print(f"\nGenerating medical notes using {max_workers} threads...")
    results = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single_note, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc="Generating") as pbar:
            for future in as_completed(futures):
                try:
                    index, generated_text, error = future.result()
                    if error:
                        print(f"\nWarning: Error generating note {index}: {error}")
                        failed_count += 1
                    else:
                        results.append({
                            'index': index,
                            'generated_text': generated_text
                        })
                    pbar.update(1)
                except Exception as e:
                    index = futures[future]
                    print(f"\nWarning: Exception generating note {index}: {e}")
                    failed_count += 1
                    pbar.update(1)
    
    # Save results
    if results:
        # Sort by index
        results.sort(key=lambda x: x['index'])
        
        # Write to CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'generated_text'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print("Generation Complete!")
        print(f"{'='*60}")
        print(f"Successfully generated: {len(results)} notes")
        print(f"Failed: {failed_count} notes")
        print(f"Results saved to: {output_csv}")
    else:
        print("\nError: No notes were generated successfully.")
        sys.exit(1)

if __name__ == "__main__":
    main()

