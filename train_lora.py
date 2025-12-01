import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer # <-- We still use SFTTrainer

# --- 1. Define Your Model (Qwen) ---
model_name = "Qwen/Qwen2.5-7B-Instruct"

# --- 2. (Unchanged) Load Dataset ---
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train") # (We use all 1000 samples)

# --- 3. (Unchanged) 4-bit Quantization Configuration ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- 4. (Key Update) Load Model and Tokenizer ---
print("--- (1/4) Loading model... ---")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True # (Qwen needs this)
)
# (We no longer need model.config.use_cache = False)

print("--- (2/4) Loading Tokenizer... ---")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # (This is still important)

# --- 5. (Key Update) LoRA Configuration (Unchanged) ---
# (This LoRAConfig is still correct)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, # (We used 16 before)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# --- 6. (Unchanged) Training Arguments ---
training_args = TrainingArguments(
    output_dir="./lora-results", # Temporary checkpoint
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=50,       # (We used 50 steps before)
    fp16=True,
    optim="paged_adamw_8bit" # (Use 8-bit optimizer to save memory)
)

# --- 7. (!! Key API Change !!) SFTTrainer ---
# This is the "new" API. It's simpler!
print("--- (3/4) Initializing SFTTrainer... ---")
trainer = SFTTrainer(
    model=model,                  # (Pass base model)
    train_dataset=dataset,
    peft_config=peft_config,      # (!! New !!) Pass LoRA config directly
    dataset_text_field="text",    # (!! New !!) Tell it which column is text
    max_seq_length=512,           # (!! New !!) Set max length
    tokenizer=tokenizer,
    args=training_args,
)

# --- 8. (Unchanged) Start Training ---
print("--- (4/4) Starting training... ---")
trainer.train()

# --- 9. (Unchanged) Save Your LoRA Weights ---
print("--- Training complete! Saving final weights... ---")
# (We save it to the familiar old location)
final_model_path = "./my-first-lora-weights"
trainer.save_model(final_model_path)

print(f"--- Success! Your LoRA weights have been saved to: {final_model_path} ---")
