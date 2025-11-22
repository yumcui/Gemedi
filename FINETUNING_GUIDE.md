# Llama 3.1 Medical Data Generator Fine-tuning Guide

This document provides a comprehensive guide for fine-tuning the Llama 3.1 8B Instruct model to generate synthetic patient medical records with Protected Health Information (PHI).


## Project Overview

### Objective
Fine-tune Llama 3.1 8B model to generate complete hospital discharge records with PHI based on simple patient metadata (gender, medical service).

### Runtime Environment
- **Cluster**: Brown University OSCAR Cluster
- **GPU**: RTX 3090 / A5000 (24GB VRAM)

**GPU Selection Rationale**:
1. Memory requirements:
   - Unquantized FP16: 8B params × 2 bytes = 16GB
   - 4-bit quantized: 8B × 0.5 bytes = 4GB
   - Backpropagation overhead: ~10GB
   - QLoRA overhead: ~2GB
   - **Total: ~20GB minimum VRAM**
2. BF16 support: Llama 3.1 uses BF16 training dtype, requiring sm_80+ compute capability (RTX 3090/A5000/A5500)

- **Job Scheduler**: Slurm

### Technology Stack
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **Frameworks**: Transformers + PEFT + TRL
- **Deployment**: Ollama + llama.cpp

---

## Environment Setup

### 1. Request GPU Node

```bash
# Interactive session (for testing)
interact -q gpu -g 1 -t 1:00:00 -m 32g

# Or submit batch job (production)
sbatch submit_gen.sh
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv ~/pytorch.venv
source ~/pytorch.venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (CUDA 12.1)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers>=4.43.0 accelerate bitsandbytes peft trl datasets
```

### 3. Clone llama.cpp (for format conversion)

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
```

### 4. HuggingFace Authentication

```bash
# Login to HuggingFace
huggingface-cli login

# Ensure you've accepted Llama 3.1 license agreement
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

## Fine-tuning Workflow

### Step 1: Data Preparation

#### Data Format
Training data uses JSONL format (Alpaca style):

```json
{
  "instruction": "Create a complete hospital discharge record with Protected Health Information (PHI) for a synthetic patient.",
  "input": "Generate discharge record for: Sex: Male, Medical Service: CARDIOLOGY",
  "output": "Name: John Doe  Unit No: 1234567\n\nAdmission Date: 2024-01-15...[complete discharge record]"
}
```

**Field Mapping**:
- `instruction`: System prompt (task description)
- `input`: User prompt (patient metadata)
- `output`: Expected model output (full discharge record)

#### Dataset Splits
```
phi_training_alpaca_train.jsonl   # Training set
phi_training_alpaca_val.jsonl     # Validation set
phi_training_alpaca_test.jsonl    # Test set
```


---

### Step 2: LoRA Fine-tuning

#### Core Scripts
- **Training script**: `train_generator.py`
- **Submit script**: `submit_gen.sh`


#### Run Training

```bash
# Submit Slurm job
sbatch submit_gen.sh

# Check job status
myq

# Monitor logs
tail -f slurm-gen-<job_id>.out

# Check GPU utilization (on compute node)
nvidia-smi
```

---

### Step 3: Model Merging

#### Why Merge?
LoRA training only saves **delta weights** (adapter). Ollama doesn't support loading adapters directly, requires full merged model.

#### Core Scripts
- **Merge script**: `merge_peft.py`
- **Submit script**: `submit_merge.sh`

#### Run Merge

```bash
sbatch submit_merge.sh
```

---

### Step 4: GGUF Conversion

#### Why GGUF?
- **GGUF** (GPT-Generated Unified Format) is llama.cpp's native format
- Supports CPU inference and dynamic quantization
- Ollama's underlying engine is llama.cpp

#### Core Scripts
- **Conversion script**: `llama.cpp/convert_hf_to_gguf.py`
- **Submit script**: `submit_convert.sh`


#### Run Conversion

```bash
# Run on CPU node (no GPU needed)
sbatch submit_convert.sh
```
---

### Step 5: Ollama Deployment

#### 5.1 Configure Storage Path

```bash
# Set model storage path (avoid Home quota limits)
export OLLAMA_MODELS="/oscar/scratch/$USER/ollama_models"
mkdir -p $OLLAMA_MODELS
```

#### 5.2 Create Modelfile

Create `Modelfile` in project root:

```dockerfile
FROM ./patient_gen_f16.gguf

SYSTEM """Create a complete hospital discharge record with Protected Health Information (PHI) for a synthetic patient."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
```

**Parameters**:
- `temperature`: Controls randomness (0.7 for diverse generation)
- `num_ctx`: Context window size

#### 5.3 Start Ollama Service

```bash
# Load Ollama module
module load ollama

# Configure concurrency
export OLLAMA_NUM_PARALLEL=4        # Support 4 concurrent requests
export OLLAMA_MAX_LOADED_MODELS=1   # Load 1 model at a time

# Start service in background
ollama serve &
```

#### 5.4 Import Model

```bash
# Create model
ollama create medical-generator -f Modelfile

# Verify model
ollama list
```

#### 5.5 Test Inference

```bash
# CLI test
ollama run medical-generator "Generate discharge record for: Sex: Male, Service: CARDIOLOGY"

# API test
curl http://localhost:11434/api/generate -d '{
  "model": "medical-generator",
  "prompt": "Generate discharge record for: Sex: Female, Service: PSYCHIATRY",
  "stream": false
}'
```
---


### Slurm Configuration

```bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -C "geforce3090|a5000"    # GPU constraint (sm_86)
#SBATCH -t 4:00:00                # Max 4 hours
#SBATCH --mem=32G                 # 32GB RAM
#SBATCH -J fake-patient-gen       # Job name
#SBATCH -o slurm-gen-%j.out       # Log output
```

---

## Troubleshooting

### Issue 1: `libcusparse.so.12` Not Found

**Error**:
```
OSError: libcusparse.so.12: cannot open shared object file
```

**Cause**: OSCAR CUDA library path not auto-loaded.

**Solution**: Add to Slurm script:

```bash
export CUDA_HOME="/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/cuda-12.1.1-ebglvvqo7uhjvhvff2qlsjtjd54louaf"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
```

---

### Issue 2: HuggingFace 401/403 Error

**Error**:
```
HTTPError: 401 Client Error: Unauthorized
```

**Causes**:
1. Not logged into HuggingFace
2. Haven't accepted Llama 3.1 license agreement

**Solution**:

```bash
# 1. Login
huggingface-cli login

# 2. Accept license
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

# 3. Verify token
huggingface-cli whoami
```

---

### Issue 3: CUDA Out of Memory (OOM)

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Cause**: Insufficient VRAM (24GB GPU).

**Solutions**:

1. **Verify configuration**:
```python
# Ensure these optimizations are enabled
gradient_checkpointing=True
per_device_train_batch_size=1  # Reduce to 1
gradient_accumulation_steps=8  # Increase accumulation
```

2. **Reduce LoRA rank**:
```python
r=32  # From 64 to 32
```

3. **Reduce sequence length**:
```python
max_seq_length=512  # From 1024 to 512
```

---

### Issue 4: Slow Ollama Inference

**Symptom**: Generation speed <5 tokens/sec

**Cause**: Running on Login node (CPU mode).

**Solution**:

```bash
# Must run on GPU node
interact -q gpu -g 1 -t 2:00:00 -m 32g

# Then start Ollama
module load ollama
ollama serve &
```

---

