# 🩺 Clinical Documentation Discriminator Fine-Tuning  
### (QLoRA + Llama-3.1-8B-Instruct)

This repository provides a full training pipeline for a **clinical documentation hallucination discriminator**, fine-tuned with **QLoRA** on Llama-3.1-8B-Instruct.

The discriminator evaluates medical discharge notes and outputs:

```json
{
  "reason": "good if no issues, otherwise describe the error in ≤50 words"
}
```

The model is trained to detect:
- Internal clinical logic errors  
- Timeline hallucinations  
- PHI inconsistencies  
- Missing justification or contradictions  
- Structural formatting violations  

---

# 🚀 1. Features

### ✔ 4-bit QLoRA (NF4)
Greatly reduces GPU memory usage.

### ✔ Chat-formatted supervised training
Uses TRL `SFTTrainer` with `apply_chat_template`.

### ✔ LoRA adapters on all attention + MLP projections
Updates q/k/v/o_proj, up/down/gate_proj.

### ✔ Fully compatible with Llama-3.1 system/user/assistant message format.

---

# 📦 2. Installation

### OSCAR cluster:
```bash
module load python/3.11.0s-ixrhc3q
module load cuda/12.1.1
```

### Create environment:
```bash
python -m venv llama_env
source llama_env/bin/activate
```

### Install Python dependencies:
```bash
pip install torch transformers datasets accelerate
pip install peft trl bitsandbytes
```

---

# 📁 3. Dataset Format

Training and evaluation datasets must be **JSONL** files containing:

```json
{"note": "<medical note text>", "reason": "<evaluation reason>"}
```

Example:

```json
{
  "note": "Name: John Doe\nAdmission Date: 2024-05-01\nDischarge Date: 2024-05-02\n...",
  "reason": "Timeline error: admission after discharge."
}
```

Empty reasons are automatically replaced with `"good"`.

---

# 🧠 4. Running the Training Script

Basic usage:

```bash
python sub_discriminator.py
```

Custom arguments:

```bash
python finetune_discriminator.py \
  --train_data train_discriminator.jsonl \
  --eval_data eval_discriminator.jsonl \
  --output_dir ./llama3-discriminator-v2 \
  --num_train_epochs 4 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16
```

---

# ⚙️ 5. SLURM Job Example (OSCAR)

Create `run_discriminator.sh`:

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C "geforce3090|a5000|a5500"
#SBATCH -t 8:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-discriminator-%j.out

module load python/3.11.0s-ixrhc3q cuda/12.1.1
source ~/Gemedi/llama_env/bin/activate

python finetune_discriminator.py
```

Submit:

```bash
sbatch run_discriminator.sh
```

Monitor:

```bash
tail -f slurm-discriminator-<jobid>.out
```

---

# 📦 6. Output Model

The fine‑tuned LoRA adapter will be saved to:

```
llama3-discriminator/
 ├── adapter_model.bin
 ├── adapter_config.json
 ├── tokenizer.json
 ├── tokenizer.model
 └── other PEFT files...
```

Load for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "./llama3-discriminator")

tokenizer = AutoTokenizer.from_pretrained(base)
```

---

# 🎉 7. Done!
Your discriminator training pipeline is ready to run with QLoRA on Llama‑3.1‑8B.
