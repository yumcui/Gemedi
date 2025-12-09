#!/usr/bin/env python3

import asyncio
import aiohttp
import csv
import json
from datetime import datetime
from typing import List, Dict
from tqdm.asyncio import tqdm

MODEL_NAME = "patient-generator"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_CONCURRENT = 4  
MAX_RETRIES = 3
TIMEOUT = 120

INPUT_CSV = "/oscar/home/rzhan221/Gemedi/both_discriminator_output.csv"
OUTPUT_CSV = "/oscar/home/rzhan221/Gemedi/corrected_medical_notes.csv"
CHECKPOINT_CSV = "/oscar/home/rzhan221/Gemedi/corrected_medical_notes_checkpoint.csv"
PROGRESS_JSON = "/oscar/home/rzhan221/Gemedi/fix_progress.json"
CHECKPOINT_INTERVAL = 10
def build_correction_prompt(medical_note: str, realism: str, medical_reasoning: str) -> str:
    from datetime import datetime

    # Build correction rules
    correction_rules = []

    if realism and realism.strip() and realism.strip().lower() != "good":
        correction_rules.append(f"- Realism Issue: {realism}")

    if medical_reasoning and medical_reasoning.strip():
        correction_rules.append(f"- Medical Reasoning Issue: {medical_reasoning}")

    if not correction_rules:
        return None

    correction_rules_text = "\n".join(correction_rules)

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are a Medical Record Editor. Your task is to correct formatting and logical errors in medical notes.

IMPORTANT DATE REQUIREMENT:
- Today's date is {today}
- All dates in the medical note (Admission Date, Discharge Date, DOB, etc.) MUST be in the PAST (before {today})
- If any date is TODAY or in the FUTURE, you MUST change it to a reasonable past date
- Ensure all dates are logically consistent (DOB < Admission < Discharge)

Input Data:
{medical_note}

Correction Rules
Must rewrite the medical record above applying the following fixes and make sure that the output does not have these issues:
{correction_rules_text}

CRITICAL DATE FIXES:
- Check ALL dates in the note
- If Admission Date >= {today}, change it to a past date (e.g., 2-4 weeks ago)
- If Discharge Date >= {today}, change it to a past date (e.g., 1-3 weeks ago)
- Ensure DOB is at least 18+ years before admission for adults, or appropriate age for pediatrics
- Never use future dates

CRITICAL OUTPUT REQUIREMENTS:
1. Output ONLY the corrected medical note
2. Do NOT add any explanations, comments, or extra text
3. Do NOT repeat sections that already exist
4. Keep the EXACT SAME structure and length as the input
5. Start with "Name:" and end with the Followup Instructions paragraph
6. STOP immediately after Followup Instructions - DO NOT continue writing

Output the corrected medical note now:"""

    return prompt

def clean_generated_note(text: str) -> str:

    lines = text.split('\n')

    followup_start = -1
    for i, line in enumerate(lines):
        if 'Followup Instructions:' in line or 'Follow-up Instructions:' in line or 'Follow up Instructions:' in line:
            followup_start = i
            break

    if followup_start == -1:
        return text


    result_lines = lines[:followup_start + 1]  

    i = followup_start + 1
    while i < len(lines):
        line = lines[i]

        if any(marker in line for marker in [
            'Discharge Instructions - ',
            'Patient Demographics:',
            'Admission Date:',
            'Service:',
            'Allergies:',
            'Attending:',
            'Discharge Medications:',
            'Discharge Diagnosis:',
            'Hospital Course:',
            '---',
            '###'
        ]):
            break


        if i > followup_start + 10 and line.strip() == '' and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and (next_line[0].isupper() or next_line.startswith('Name:')):
                break

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines).strip()

async def generate_corrected_note(
    session: aiohttp.ClientSession,
    prompt: str,
    idx: int,
    semaphore: asyncio.Semaphore,
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
                        "temperature": 0.7,
                        "num_predict": 2048,
                        "top_p": 0.9,
                        "stop": [
                            "\n\nDischarge Instructions - Ortho:",
                            "\n\nAdditional",
                            "\n\n---",
                            "###",
                            "Note:",
                            "\n\nPatient Demographics:",
                            "\n\nService:"
                        ]
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

           
                        cleaned_text = clean_generated_note(generated_text)

                        return {
                            "id": idx,
                            "corrected_note": cleaned_text,
                            "status": "corrected"
                        }
                    else:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {
                            "id": idx,
                            "corrected_note": "",
                            "status": f"error: HTTP {response.status}"
                        }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "id": idx,
                    "corrected_note": "",
                    "status": f"error: {str(e)}"
                }

        return {
            "id": idx,
            "corrected_note": "",
            "status": "error: Max retries exceeded"
        }

async def process_batch(
    records: List[Dict],
    max_concurrent: int,
    model_name: str
) -> List[Dict]:

    tasks_to_run = []
    results = []

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        for idx, record in enumerate(records):
            medical_note = record['medical_note']
            realism = record.get('realism_reasoning', '')
            medical_reasoning = record.get('medical_reasoning', '')

            prompt = build_correction_prompt(medical_note, realism, medical_reasoning)

            if prompt is None:
                results.append({
                    'id': idx,
                    'original_note': medical_note,
                    'realism_issue': realism,
                    'medical_reasoning_issue': medical_reasoning,
                    'corrected_note': medical_note,
                    'status': 'no_issues'
                })
            else:
                task = generate_corrected_note(session, prompt, idx, semaphore, model_name)
                tasks_to_run.append((idx, record, task))

        if tasks_to_run:
            pbar = tqdm(total=len(tasks_to_run), desc="process")

            for idx, record, task in tasks_to_run:
                result = await task
                pbar.update(1)

                results.append({
                    'id': result['id'],
                    'original_note': record['medical_note'],
                    'realism_issue': record.get('realism_reasoning', ''),
                    'medical_reasoning_issue': record.get('medical_reasoning', ''),
                    'corrected_note': result['corrected_note'] if result['status'] == 'corrected' else record['medical_note'],
                    'status': result['status']
                })

            pbar.close()


    results.sort(key=lambda x: x['id'])

    return results

def save_results(results: List[Dict], filename: str):

    with open(filename, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['original_note', 'realism_issue', 'medical_reasoning_issue', 'corrected_note', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'original_note': r['original_note'],
                'realism_issue': r['realism_issue'],
                'medical_reasoning_issue': r['medical_reasoning_issue'],
                'corrected_note': r['corrected_note'],
                'status': r['status']
            })

def save_progress(stats: Dict):

    with open(PROGRESS_JSON, 'w') as f:
        json.dump(stats, f, indent=2)


def main():

    records = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    total_records = len(records)

    stats = {
        'total': total_records,
        'corrected': 0,
        'no_issues': 0,
        'errors': 0,
        'start_time': datetime.now().isoformat()
    }

    try:
        results = asyncio.run(process_batch(records, MAX_CONCURRENT, MODEL_NAME))

        for r in results:
            if r['status'] == 'corrected':
                stats['corrected'] += 1
            elif r['status'] == 'no_issues':
                stats['no_issues'] += 1
            else:
                stats['errors'] += 1

        stats['end_time'] = datetime.now().isoformat()

        
        save_results(results, OUTPUT_CSV)
        save_progress(stats)




if __name__ == "__main__":
    main()
