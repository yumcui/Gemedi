# Medical Notes Evaluation Pipeline

This script evaluates medical notes from a CSV file using two evaluation methods:

1. **Realism Score**: Evaluates text quality using integrated realism logic (document structure, linguistic, and clinical consistency)
2. **Gemini Medical Reasoning**: Uses Google Gemini API to evaluate medical reasoning based on audit criteria

The script is completely standalone and does not require any external module imports.

## Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install pandas google-generativeai tqdm python-dateutil
  ```
  ```bash
  pip install ollama
  ```
- Google Gemini API key (set as environment variable)

## Usage

### Option 1: SLURM Cluster (Recommended for Oscar)

1. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```

2. Submit evaluation job with `sbatch`:
   ```bash
   sbatch submit_evaluation.sh <input_csv_file> [output_csv_file] [max_workers] [num_rows]
   ```

   Examples:
   ```bash
   # Basic usage
   sbatch submit_evaluation.sh llama3.1_one_shot/generated_medical_notes.csv
   
   # Specify output file
   sbatch submit_evaluation.sh input.csv output.csv
   
   # Specify threads and limit rows
   sbatch submit_evaluation.sh input.csv output.csv 2 50
   ```

   The script automatically:
   - Loads Ollama module and starts server
   - Activates Python environment
   - Runs evaluation with all arguments
   - Stops Ollama server after completion

### Option 2: Direct Python Execution

1. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```

2. Start Ollama server (required for clinical consistency evaluation):
   ```bash
   module load ollama  # On Oscar cluster
   ollama serve &      # Start in background
   ```
   
   **Note**: Without Ollama, the script will skip clinical consistency checks but still run other evaluations.

3. Run the evaluation script:
   ```bash
   python evaluate_medical_notes.py <input_csv_file> [output_csv_file] [max_workers] [num_rows]
   ```

   Examples:
   ```bash
   # Basic usage (default: 2 threads)
   python3 evaluate_medical_notes.py llama3.1_one_shot/generated_medical_notes.csv  
   
   # Specify output file
   python evaluate_medical_notes.py input.csv output.csv
   
   # Specify threads and limit rows
   python evaluate_medical_notes.py input.csv output.csv 2 50
   ```

### Arguments

   - `input_csv_file`: Path to input CSV file (required)
   - `output_csv_file`: Path to output CSV file (optional, default: `<input>_evaluated.csv`)
   - `max_workers`: Number of threads for parallel processing (optional, default: 2)
   - `num_rows`: Number of rows to process (optional, default: all rows)

## Input Format

The CSV file must contain a column named `generated_text` with the medical notes to evaluate.

**Auto-column detection**: The script automatically handles different column names:
- If CSV has only one column, it will be renamed to `generated_text`
- If `generated_text` is not found, the script will look for similar names like `generated_note`, `generated`, `text`, etc.

## Output

The script adds four columns to the CSV:

- `realism_score`: Realism evaluation score (0-1, weighted 0.5)
- `gemini_medical_reasoning_evaluation`: Gemini API evaluation score (0-1, weighted 0.5)
- `overall_score`: Combined score = (realism_score × 0.5) + (gemini_score × 0.5)
- `realism_issues`: Detailed information about realism evaluation issues (linguistic, structure, clinical consistency)

After evaluation, the script prints average scores for all three metrics.

## Performance Tips

- **Rate Limiting**: The script includes automatic rate limiting for Gemini API
  - Free tier limit: 10 requests per minute per model
  - Default: 2 threads (safe for free tier)
  - Rate limiter automatically waits when approaching limits
  - Automatic retry with exponential backoff on 429 errors
- **Threading**: 
  - For free tier: Use 1-2 threads (default: 2)
  - For paid tier: Can increase to 5-10 threads for faster processing
- **Speed**: With rate limiting enabled, processing is slower but avoids quota errors

