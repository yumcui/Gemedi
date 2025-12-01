# Quick Start Guide

## System Overview

This system includes:

1. **Data Preparation**: Convert CSV to training format
2. **Generator Training**: Train generator to create medical records
3. **Discriminator Training**: Train discriminator to evaluate sample quality

## Quick Start (2 Steps)

### Step 1: Prepare Discriminator Training Data

```bash
cd /users/zzhou190/projects/main/Gemedi
python prepare_discriminator_data.py
```

This creates `discriminator_train.jsonl` (~2000 samples: 1000 good + 1000 bad)

### Step 2: Train Models

#### 2.1 Train Generator

```bash
python train_generator.py
# or
sbatch submit_gen.sh
```

#### 2.2 Train Discriminator

```bash
python train_discriminator.py
# or
sbatch submit_discriminator.sh
```

## File Structure

```
Gemedi/
├── prepare_discriminator_data.py    # Data preparation
├── train_generator.py                # Generator training
├── train_discriminator.py            # Discriminator training
├── test_generator.py                 # Test generator
├── test_discriminator.py             # Test discriminator
├── test_both.py                      # Test both models
├── submit_gen.sh                     # Slurm script for generator
└── submit_discriminator.sh           # Slurm script for discriminator
```

## Testing

After training, test the models:

```bash
# Test generator
python test_generator.py

# Test discriminator
python test_discriminator.py

# Test complete workflow
python test_both.py
```

## Output

- **Generator**: `llama3-synthetic-patient-generator/`
- **Discriminator**: `llama3-phi-discriminator/`

## Notes

1. **Memory Requirements**: Need sufficient GPU memory (recommended 24GB+)
2. **Data Paths**: Ensure CSV file paths are correct
3. **Model Paths**: Ensure model paths are correct
