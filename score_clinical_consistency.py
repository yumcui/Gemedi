import pandas as pd
import re
import time
import ollama
from enum import Enum
from datetime import datetime
from dateutil import parser
from typing import Optional, Tuple
import concurrent.futures
import threading

class ConsistencyCheck(Enum):
    AGE = "age"
    SEX = "biological sex"

# Configuration
MODEL_NAME = "llama3.1" # "llama3.1:70b"
MAX_WORKERS = 7
AGE_WEIGHT = 0.5
SEX_WEIGHT = 0.5
USE_FULL_TEXT = True

class ClinicalConsistencyEvaluator:
    @staticmethod
    def get_relevance(check_type: ConsistencyCheck, value: str, diagnosis: str, clinical_text: str, retries: int = 2) -> int:
        """
        Generic method to score clinical consistency for different attributes.

        :param check_type: Use ConsistencyCheck.AGE or ConsistencyCheck.SEX
        :param value: The value of patient's attribute
        :param full_text: The full discharge summary string.
        # :param context: Optional additional context for the model (medical history, complications ...)
        :param retries: The number of retries allowed
        :return: An integer score between 1 and 10. -1 to flag an error
        """
        if check_type == ConsistencyCheck.AGE:
            criteria = (
                "epidemiological likelihood. Check if the Primary Diagnosis and the patient's medical history "
                "are plausible for a patient of this age."
            )
            if value == "-1":
                print(f"Invalid age: {value}.")
                return -1
        elif check_type == ConsistencyCheck.SEX:
            criteria = (
                "anatomical compatibility and biological sex predilection. "
                "Check if the Primary Diagnosis is consistent with the patient's biological sex."
            )
            if value == None:
                print(f"Invalid sex: {value}.")
                return -1
        else:
            raise ValueError("Invalid ConsistencyCheck type")
        text_clean = " ".join(clinical_text.split())
        truncated_text = text_clean[:2000] + "..." if len(text_clean) > 2000 else text_clean
        if not USE_FULL_TEXT:
           truncated_text = "" 
        prompt = (
            f"You are an expert medical auditor. Evaluate the clinical plausibility of a diagnosis.\n\n"
            "TASK:\n"
            f"Patient {check_type.value.title()}: {value}\n"
            f"Primary Diagnosis: {diagnosis}\n"
            f"Discharge Summary Snippet: {truncated_text}\n\n"
            "INSTRUCTIONS:\n"
            f"- Evaluate strictly based on {criteria}.\n"
            "- Rate consistency on a scale of 1-10 (1 = Impossible/Highly Unlikely, 10 = Very Common/Consistent).\n"
            "- Do not provide explanations, notes, or labels (e.g., do not write 'Score:').\n"
            "- Return ONLY the integer.\n\n"
            "SCORE:"
        )
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(retries + 1):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    options={"temperature": 0.2, "num_predict": 10}
                )
                score_str = response["message"]["content"].strip()
                match = re.search(r"\b([1-9]|10)\b", score_str)
                if match:
                    return int(match.group(1))
                # If parsing fails, append the model's wrong answer and a correction instruction
                # for the next loop iteration.
                print(f"Attempt {attempt+1} failed to parse: {score_str}. Retrying {check_type.value} check with correction...")
                messages.append({"role": "assistant", "content": score_str})
                messages.append({"role": "user", "content": "You did not output an integer. Please output ONLY the integer score (1-10)."})
            except Exception as e:
                print(f"API Error on {check_type.value} check attempt {attempt+1}: {e}")
                time.sleep(2) # Simple wait for network errors
        # Return -1 to indicate an error; handle it later manually.
        return -1
        
# --- Extraction Helpers ---

def calculate_age_at_discharge(text: str) -> int:
    """
    Parses the Discharge Date and Date of Birth from a clinical note text
    and calculates the patient's age at the time of discharge.

    :param text: The raw clinical note text.
    :return: The calculated age in years (int). Returns -1 if dates cannot be found or parsed.
    """
    
    # 1. Define Regex Patterns
    # We use case-insensitive matching and allow for flexible spacing.
    # We look for the label, followed by an optional colon, optional whitespace, and then capture the date string.
    # The date regex is broad to catch YYYY-MM-DD, MM/DD/YYYY, etc., then we validate with dateutil.
    
    # Pattern for Discharge Date
    # Matches "Discharge Date:" followed by non-newline characters until end of line or a large gap
    discharge_pattern = r"Discharge Date:\s*([0-9\-\/]+)"
    # Pattern for Date of Birth
    # Matches "Date of Birth:" or "DOB:" followed by non-newline characters
    dob_pattern = r"(?:Date of Birth|DOB):\s*([0-9\-\/]+)"
    try:
        # 2. Search for the dates
        discharge_match = re.search(discharge_pattern, text, re.IGNORECASE)
        dob_match = re.search(dob_pattern, text, re.IGNORECASE)
        if not discharge_match:
            print("Error: Could not find 'Discharge Date' in text.")
            return -1
        if not dob_match:
            print("Error: Could not find 'Date of Birth' in text.")
            return -1
        # 3. Extract and Clean Strings
        discharge_str = discharge_match.group(1).strip()
        dob_str = dob_match.group(1).strip()
        # 4. Parse Dates
        discharge_date = parser.parse(discharge_str)
        dob_date = parser.parse(dob_str)
        # 5. Calculate Age
        # Standard age formula: Difference in years, minus 1 if the birthday hasn't happened yet this year
        age = discharge_date.year - dob_date.year - (
            (discharge_date.month, discharge_date.day) < (dob_date.month, dob_date.day)
        )
        return age
    except (ValueError, OverflowError) as e:
        print(f"Error parsing dates: {e}")
        return -1

def extract_sex(text: str) -> Optional[str]:
    """
    Extracts the sex/gender from a clinical note text.

    :param text: The raw clinical note text.
    :return: The extracted sex string (e.g., 'F', 'M', 'Female'). Returns None if not found.
    """
    # Pattern for Sex
    # Matches "Sex:" or "Gender:" followed by optional whitespace and captures word characters
    # We use [A-Za-z]+ to capture "F", "M", "Female", "Male"
    sex_pattern = r"(?:Sex|Gender):\s*([A-Za-z]+)"
    match = re.search(sex_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    print("Error: Could not find 'Sex' or 'Gender' in text.")
    return None

def extract_diagnosis(text: str) -> Optional[str]:
    pattern = r"Discharge Diagnosis:\s*([\s\S]+?)(?=\n\s*\n|Discharge Condition:|History|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace('\n', ' ')
    return "Unknown Diagnosis" # Fallback string so the prompt doesn't break

# --- Processing Logic ---

def process_single_row(row_data: Tuple[int, str]) -> dict:
    """
    Worker function to process a single row.
    Accepts a tuple (index, text) to keep track of row numbers.
    """
    index, text = row_data
    result = {
        "index": index,
        "age_extracted": -1,
        "sex_extracted": None,
        "diagnosis_extracted": None,
        "score_age": -1,
        "score_sex": -1,
        "weighted_score": -1,
        "error": None
    }
    if not isinstance(text, str):
        result["error"] = "Invalid text format"
        return result
    # 1. Extract Data
    age = calculate_age_at_discharge(text)
    sex = extract_sex(text)
    diagnosis = extract_diagnosis(text) 
    result["age_extracted"] = age
    result["sex_extracted"] = sex
    result["diagnosis_extracted"] = diagnosis
    # 2. Validate Extraction
    if age == -1 or not sex or not diagnosis:
        result["error"] = "Extraction failed (Missing Date, Sex, Diagnosis)"
        return result
    # 3. LLM Evaluation
    try:
        age_score = ClinicalConsistencyEvaluator.get_relevance(
            ConsistencyCheck.AGE, 
            value=str(age), 
            diagnosis=diagnosis, 
            clinical_text=text
        )
        sex_score = ClinicalConsistencyEvaluator.get_relevance(
            ConsistencyCheck.SEX, 
            value=sex, 
            diagnosis=diagnosis, 
            clinical_text=text
        )
        result["score_age"] = age_score
        result["score_sex"] = sex_score
        # 4. Compute Weighted Score
        if age_score != -1 and sex_score != -1:
            final = (age_score * AGE_WEIGHT) + (sex_score * SEX_WEIGHT)
            result["weighted_score"] = round(final, 2)
        else:
            result["error"] = "LLM returned -1 (API failure)"
    except Exception as e:
        result["error"] = str(e)
    return result

def process_csv(input_path: str, output_path: str, max_rows: Optional[int] = None):
    """
    Orchestrates the CSV loading, processing, and saving.
    """
    print(f"Loading {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return
    if "filled_text" not in df.columns:
        print("Error: Column 'filled_text' not found in CSV.")
        return
    # Apply max_rows limit if provided
    if max_rows is not None:
        print(f"TEST: Limiting processing to first {max_rows} rows.")
        df = df.head(max_rows)
    print(f"Processing {len(df)} records with {MAX_WORKERS} threads...")
    # Prepare data for the executor (Index, Text) tuples
    rows_to_process = list(zip(df.index, df["filled_text"]))
    results_list = []
    # Start Timer
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit tasks
        future_to_row = {executor.submit(process_single_row, row): row for row in rows_to_process}
        # process as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_row)):
            data = future.result()
            results_list.append(data)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(df)} records...")

    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
    # --- Merge Results back to DataFrame ---
    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        results_df.set_index("index", inplace=True)
        final_df = df.merge(results_df, left_index=True, right_index=True)
    else:
        final_df = df
        print("Warning: No results generated.")
    # Save to csv
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    # Show sample
    if not final_df.empty and "weighted_score" in final_df.columns:
        print("\n--- Sample Output ---")
        cols = ["age_extracted", "sex_extracted", "text", "score_age", "score_sex", "weighted_score"]
        print(final_df[[c for c in cols if c in final_df.columns]].head())

def main():
    INPUT_CSV = "patient_extracted_with_synth_phi_llm_refined.csv"
    OUTPUT_CSV = "patient_notes_clinical_consistency_scored.csv"
    # Call the processor with global constants
    # Set max_rows=None for full run
    process_csv(INPUT_CSV, OUTPUT_CSV)

if __name__ == "__main__":
    main()