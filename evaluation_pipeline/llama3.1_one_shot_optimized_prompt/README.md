# Llama 3.1 Optimized Prompt Medical Notes Generation

This directory contains scripts to generate medical notes using Llama 3.1 model with optimized prompts that combine:
1. Service-specific biological constraints (SERVICE_RULES)
2. Chain-of-thought reasoning prompts
3. One-shot learning with a high-quality example

## Features

- **Biological Validity**: Uses SERVICE_RULES to ensure age/sex/service combinations are realistic
- **Chain-of-Thought**: Prompts include reasoning steps for diagnosis selection, treatment planning, and dosage calculation
- **Strict Medication Formatting**: Enforces complete medication signatures (Drug + Strength + Route + Frequency)
- **One-Shot Learning**: Uses a high-quality example to guide generation format and style
- **Parallel Processing**: Multi-threaded generation for faster processing

## Service Rules

The script uses predefined rules for different medical services:

- **OBSTETRICS**: Female only, age 18-45
- **PEDIATRICS**: Both sexes, age 0-17
- **UROLOGY**: Male 80%, Female 20%, age 40-85
- **CARDIOLOGY**: Both sexes, age 45-90
- **ORTHOPEDICS**: Both sexes, age 18-85
- **NEUROLOGY**: Both sexes, age 25-90
- **GENERAL_MEDICINE**: Both sexes, age 20-85

## Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install ollama tqdm
  ```
- Ollama installed and running
- Llama 3.1 model installed:
  ```bash
  ollama pull llama3.1
  ```

## Usage

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Run the generation script:
   ```bash
   python generate_optimized_notes.py [output_csv_file] [num_notes] [max_workers]
   ```

   Examples:
   ```bash
   # Generate 50 notes with default settings
   python generate_optimized_notes.py
   
   # Generate 50 notes with custom output file
   python generate_optimized_notes.py my_notes.csv
   
   # Generate 100 notes
   python generate_optimized_notes.py my_notes.csv 100
   
   # Generate with 10 parallel threads
   python generate_optimized_notes.py my_notes.csv 50 10
   ```

## Arguments

- `output_csv_file`: Path to output CSV file (optional, default: `generated_medical_notes.csv`)
- `num_notes`: Number of notes to generate (optional, default: 50)
- `max_workers`: Number of threads for parallel processing (optional, default: 5)

## Output Format

The script generates a CSV file with the following columns:
- `index`: Index of the generated note
- `generated_text`: The generated medical discharge summary

## Prompt Features

The optimized prompt includes:

1. **Patient Profile**: Service, Sex, Age, DOB
2. **Chain-of-Thought Instructions**:
   - Diagnosis selection based on Age/Sex/Service
   - Treatment plan identification
   - Dosage calculation (pediatric vs adult)
   - Safety checks
3. **Medication Constraints**:
   - No ranges (e.g., "1-2 tablets" → "1 tablet" or "2 tablets")
   - Complete signatures (Drug + Strength + Route + Frequency)
   - Consistency with diagnosis
4. **One-Shot Example**: High-quality example from `sample_text.txt`

## Notes

- The script uses temperature 0.7 for moderate creativity while maintaining quality
- Each note generation allows up to 2000 tokens
- Failed generations are skipped but counted in the summary
- Adjust `max_workers` based on your system's capabilities and Ollama's performance

