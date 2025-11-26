## Requirements

- Python 3.9+
- Required packages:
  ```bash
  pip install faker pandas ollama
  ```
- Ollama with a compatible model (default: `llama3.1`)
  - Install Ollama from https://ollama.com/download
  - Pull the model: `ollama pull llama3.1`

## Usage

### Basic Usage

```bash
python fill_phi_data_updated.py
```

This will:
1. Read from `patient_extracted.csv`
2. Process all non-empty rows
3. Generate synthetic PHI values
4. Save to `patient_extracted_with_synth_phi_llm_refined.csv`

