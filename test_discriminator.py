"""
Test Discriminator model
Evaluate quality (realism and difficulty) of medical records
"""
import torch
from transformers import AutoTokenizer
from train_discriminator import DiscriminatorModel
import os

# --- Configuration ---
DISCRIMINATOR_PATH = "./llama3-phi-discriminator"
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def load_discriminator(model_path, base_model_name):
    """Load discriminator model"""
    print(f"Loading discriminator from {model_path}...")
    
    from transformers import BitsAndBytesConfig
    from train_discriminator import DiscriminatorModel
    from peft import PeftModel
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Create Discriminator model
    model = DiscriminatorModel(base_model_name, num_labels=1, quantization_config=bnb_config)
    
    # Check if it's a LoRA adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Detected LoRA adapter, loading...")
        # Load LoRA adapter to base_model
        model.base_model = PeftModel.from_pretrained(model.base_model, model_path)
        print("LoRA adapter loaded")
    else:
        print("No LoRA adapter found, using base model only")
    
    # Load classifier weights (if saved separately)
    classifier_path = os.path.join(model_path, "classifier.pt")
    if os.path.exists(classifier_path):
        print("Loading classifier weights...")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        # Move to correct device
        device = next(model.base_model.parameters()).device
        dtype = next(model.base_model.parameters()).dtype
        model.classifier = model.classifier.to(device=device, dtype=dtype)
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def evaluate_text(model, tokenizer, text):
    """Evaluate a single text"""
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Evaluate
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits and convert to realism score
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            realism_score = torch.sigmoid(logits).item()
        elif hasattr(outputs, 'realism_score'):
            realism_score = outputs.realism_score.item()
        else:
            # If neither exists, try to calculate from loss
            raise ValueError("Cannot extract realism score from model output")
    
    # Calculate difficulty (simple mapping)
    difficulty_score = 1.0 - realism_score
    
    return {
        "realism": realism_score,
        "difficulty": difficulty_score,
        "quality": "good" if realism_score >= 0.8 else "bad"
    }

def main():
    """Main function"""
    print("=" * 60)
    print("Discriminator Inference Test")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_discriminator(DISCRIMINATOR_PATH, BASE_MODEL_NAME)
    
    # Test texts (good and bad data examples)
    test_texts = [
        # Good data example
        """Name:  Margaret Johnson                     Unit No:   1419610

Admission Date:  2024-06-01              Discharge Date:   2024-06-25

Date of Birth:  1951-03-09             Sex:   F

Service: MEDICINE

Allergies: 
No Known Allergies / Adverse Drug Reactions

Attending: Dr. Julianne Marquez""",
        
        # Bad data example (incomplete name)
        """Name:  Williams                     Unit No:   2462749

Admission Date:  2024-01-31              Discharge Date:   2024-01-22

Date of Birth:  1964-08-14             Sex:   F

Service: MEDICINE

Allergies: 
No Known Allergies / Adverse Drug Reactions

Attending: Dr. Daniel Santana""",
    ]
    
    print("\n" + "=" * 60)
    print("Evaluating samples...")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Text preview: {text[:100]}...")
        print("\nEvaluation:")
        print("-" * 60)
        
        scores = evaluate_text(model, tokenizer, text)
        
        print(f"Realism Score: {scores['realism']:.4f}")
        print(f"Difficulty Score: {scores['difficulty']:.4f}")
        print(f"Quality: {scores['quality']}")
        print("-" * 60)
    
    # If new samples are generated, can also evaluate them
    print("\n" + "=" * 60)
    print("To evaluate generated samples:")
    print("  python test_discriminator.py --text_file generated_sample_1.txt")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, help="Path to text file to evaluate")
    parser.add_argument("--text", type=str, help="Text to evaluate directly")
    args = parser.parse_args()
    
    if args.text_file:
        # Read from file
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        model, tokenizer = load_discriminator(DISCRIMINATOR_PATH, BASE_MODEL_NAME)
        scores = evaluate_text(model, tokenizer, text)
        print(f"\nRealism: {scores['realism']:.4f}")
        print(f"Difficulty: {scores['difficulty']:.4f}")
        print(f"Quality: {scores['quality']}")
    elif args.text:
        # Evaluate text directly
        model, tokenizer = load_discriminator(DISCRIMINATOR_PATH, BASE_MODEL_NAME)
        scores = evaluate_text(model, tokenizer, args.text)
        print(f"\nRealism: {scores['realism']:.4f}")
        print(f"Difficulty: {scores['difficulty']:.4f}")
        print(f"Quality: {scores['quality']}")
    else:
        # Run default test
        main()
