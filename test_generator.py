"""
Test Generator model
Generate medical record samples
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import os

# --- Configuration ---
GENERATOR_PATH = "./llama3-synthetic-patient-generator"
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def load_generator(model_path, base_model_name):
    """Load generator model"""
    print(f"Loading generator from {model_path}...")
    
    # Check if it's a LoRA adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Detected LoRA adapter, loading base model first...")
        # Load base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        print("LoRA adapter loaded successfully")
    else:
        # Full model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def generate_sample(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7):
    """Generate a single sample"""
    # Build complete prompt
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Create a complete hospital discharge record with Protected Health Information (PHI) for a synthetic patient.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode (only take newly generated part)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text

def main():
    """Main function"""
    print("=" * 60)
    print("Generator Inference Test")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_generator(GENERATOR_PATH, BASE_MODEL_NAME)
    
    # Test prompts
    test_prompts = [
        "Generate discharge record for: Sex: Male, Medical Service: MEDICINE",
        "Generate discharge record for: Sex: Female, Medical Service: CARDIOLOGY",
        "Generate discharge record for: Sex: Male, Medical Service: PSYCHIATRY",
    ]
    
    print("\n" + "=" * 60)
    print("Generating samples...")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {prompt}")
        print("\nGenerated text:")
        print("-" * 60)
        
        generated = generate_sample(model, tokenizer, prompt)
        print(generated)
        print("-" * 60)
        
        # Save to file
        output_file = f"generated_sample_{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write("Generated text:\n")
            f.write(generated)
        print(f"Saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()


