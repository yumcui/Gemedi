## Step 1: Prepare Discriminator Training Data

### Prerequisites
- `preprocess_data_label_good_phi.csv` - Good data CSV file
- `preprocess_data_label_bad_phi.csv` - Bad data CSV file

These CSV files should contain:
- `filled_text`: Medical record text
- `realism`: Realism score (0-1)
- `overall`: Detailed error information
- `linguistic_realism`, `clinical_consistency_realism`, `document_structure_realism`: Dimension scores
- `DIFFICULTY`: Difficulty score

### Run Data Preparation

```bash
cd /users/zzhou190/projects/main/Gemedi
python prepare_discriminator_data.py
```

**Note**: The script will automatically search for CSV files in:
1. Current directory
2. Parent directory (`../`)
3. `../generate_data/generate_and_preprocess/`

It will create `discriminator_train.jsonl` with ~2000 samples (1000 good + 1000 bad).

## Step 2: Train Discriminator

### Option 1: Direct Execution
```bash
python train_discriminator.py
```

### Option 2: Submit to Slurm
```bash
sbatch submit_discriminator.sh
```

### Training Configuration
- **Base Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Output**: `llama3-phi-discriminator/`
- **Epochs**: 3
- **LoRA**: r=32, alpha=16
- **Batch Size**: 1 (with gradient accumulation 8)
- **Max Length**: 512

### Expected Output
- Trained model saved to `llama3-phi-discriminator/`
- Contains LoRA adapter weights and tokenizer

## Verification

After training, you can test the discriminator:
```bash
python test_discriminator.py
```

Or test both generator and discriminator:
```bash
python test_both.py
```