# PHI Data Generation and Preprocessing Pipeline

This repository contains tools for generating and preprocessing Protected Health Information (PHI) data for training purposes. The pipeline includes scripts for generating both high-quality and low-quality (bad) data, preprocessing the data, and comparing results.

## Overview

The pipeline consists of four main scripts:

1. **fill_phi_data.py** - Generates high-quality PHI data using Faker and optional LLM enhancement
2. **fill_phi_bad_data.py** - Generates intentionally problematic PHI data for negative training samples
3. **preprocess.py** - Unified preprocessing script that evaluates data quality across three dimensions
4. **compare_good_bad_data.py** - Comparison tool to verify bad data generation

## Prerequisites

- Python 3.7+
- Required packages:
  ```bash
  pip install faker ollama csv json re datetime dateutil
  ```

- For LLM features (optional):
  - Ollama installed and running
  - A compatible LLM model (default: llama3.1)

```bash
# 1. Generate good data
python3 fill_phi_data.py
# Output: patient_extracted_with_goo_phi.csv

# 2. Generate bad data
python3 fill_phi_bad_data.py
# Output: patient_extracted_with_bad_phi.csv

# 3. Preprocess bad data
python3 preprocess.py
# Output: preprocess_data_label_bad_phi.csv

# 4. Compare results
python3 compare_good_bad_data.py patient_extracted_with_goo_phi.csv patient_extracted_with_bad_phi.csv 
```


