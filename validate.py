import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --- 1. Definition ---
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# (Important!) This points to the "stronger" weights you trained with 50 steps
adapter_path = "./my-first-lora-weights" 

# --- 2. Load Configuration ---
print("--- Loading 4-bit quantization configuration... ---")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- 3. Load Base Model and Tokenizer ---
print(f"--- Loading base model ({base_model_name})... ---")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- 4. Prepare Prompt ---
# (We use a slightly creative question to make style changes more obvious)
prompt = "<|user|>\nWrite a 4-line poem about a robot learning to code.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n" + "="*40)
print("  TEST A: Base Model (Original Llama) Response:")
print("="*40 + "\n")

# --- 5. (A) Run "Base Model" ---
base_model.config.use_cache = True # Ensure it can generate normally
outputs_base = base_model.generate(
    **inputs, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7, # We use a fixed temperature
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(outputs_base[0], skip_special_tokens=True))


# --- 6. (Important!) Load Your LoRA Weights ---
print("\n" + "="*40)
print(f"--- Loading your LoRA weights (from {adapter_path})... ---")

# (This modifies base_model, turning it into the "fine-tuned" model)
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
peft_model.config.use_cache = True

print("--- Weights loaded successfully! ---")
print("\n" + "="*40)
print("  TEST B: Your Fine-tuned Model (Llama + 50 steps) Response:")
print("="*40 + "\n")

# --- 7. (B) Run "Fine-tuned Model" ---
outputs_peft = peft_model.generate(
    **inputs, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7, # (Using exactly the same parameters)
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(outputs_peft[0], skip_special_tokens=True))
print("\n" + "="*40)
print("--- Comparison complete! ---")
