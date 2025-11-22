from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Path configuration
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH = "./llama3-synthetic-patient-generator"
OUTPUT_PATH = "./merged_llama3_patient_gen"

print(f"Step 1: Loading base model {BASE_MODEL_ID} ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print(f"Step 2: Loading LoRA weights from {LORA_PATH} ...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Step 3: Merging and unloading adapter...")
model = model.merge_and_unload()

print(f"Step 4: Saving merged model to {OUTPUT_PATH} ...")
model.save_pretrained(OUTPUT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Merge completed! You can now convert it to GGUF using llama.cpp.")
