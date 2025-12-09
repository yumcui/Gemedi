#!/usr/bin/env python3

import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm
import json
import random
import datetime
from typing import List, Dict
import argparse

MODEL_NAME = "patient-generator" 
OLLAMA_URL = "http://localhost:11434/api/generate"
TOTAL_RECORDS = 25
MAX_CONCURRENT = 4
MAX_RETRIES = 3     
TIMEOUT = 120       


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

def get_random_dob(age: int) -> str:
    today = datetime.date.today()
    birth_year = today.year - age
    start_date = datetime.date(birth_year, 1, 1)
    end_date = datetime.date(birth_year, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    dob = start_date + datetime.timedelta(days=random_days)
    return dob.strftime("%Y-%m-%d")

def construct_detailed_prompt(sex: str, service: str, age: int, dob: str) -> str:

    is_pediatric = "YES" if age < 18 else "NO"
    
    return f"""
### ROLE:
You are an expert Clinical Pharmacist and Medical Scribe. You prize accuracy, standard of care, and specific dosing.

### PATIENT PROFILE:
- **Service**: {service}
- **Sex**: {sex}
- **Age**: {age} (Pediatric Case: {is_pediatric})
- **DOB**: {dob}

### INSTRUCTION (CHAIN OF THOUGHT):
Before generating the note, perform the following reasoning steps implicitly:
1. **Diagnosis Selection**: Choose a common, realistic diagnosis for this specific Age/Sex/Service.
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

### OUTPUT:
Generate the medical note now.
"""

def generate_biologically_valid_prompts(num_records: int) -> List[str]:
    prompts = []
    service_keys = list(SERVICE_RULES.keys())
    
    for _ in range(num_records):
        service = random.choice(service_keys)
        rule = SERVICE_RULES[service]
        
        sex = random.choices(
            list(rule["sex_weights"].keys()), 
            weights=list(rule["sex_weights"].values())
        )[0]
        
        age = random.randint(rule["age_min"], rule["age_max"])
        dob = get_random_dob(age)
        
        prompt = construct_detailed_prompt(sex, service, age, dob)
        prompts.append(prompt)
        
    return prompts


async def generate_single_record(
    session: aiohttp.ClientSession,
    prompt: str,
    idx: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    model_name: str
) -> Dict:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5, 
                        "num_predict": 1024,
                        "stop": ["### ROLE", "User:"] 
                    }
                }
                async with session.post(
                    OLLAMA_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        generated_text = result.get("response", "").strip()
                        pbar.update(1)
                        return {
                            "id": idx,
                            "generated_note": generated_text,
                            "status": "success"
                        }
                    else:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {"id": idx, "generated_note": f"HTTP {response.status}", "status": "failed"}
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"id": idx, "generated_note": str(e), "status": "failed"}

    return {"id": idx, "generated_note": "Unknown Error", "status": "failed"}

async def generate_all_records(prompts: List[str], max_concurrent: int, model_name: str) -> List[Dict]:

    
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(prompts), desc="generate processing")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_single_record(session, prompt, idx, semaphore, pbar, model_name)
            for idx, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
    
    pbar.close()
    return results


def save_to_csv(records: List[Dict], filename: str):
    success_records = [r for r in records if r["status"] == "success"]
    failed_count = len(records) - len(success_records)
    
    if not success_records:
        return


    df = pd.DataFrame({
        "note": [r["generated_note"] for r in success_records]
    })
    

    df.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=TOTAL_RECORDS)
    parser.add_argument("-c", "--concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("-o", "--output", type=str, default="generated_medical_notes_cot.csv",)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    
    prompts = generate_biologically_valid_prompts(args.num)
    

    try:
        results = asyncio.run(generate_all_records(prompts, args.concurrent, args.model))
        save_to_csv(results, args.output)
    except KeyboardInterrupt:


if __name__ == "__main__":
    main()