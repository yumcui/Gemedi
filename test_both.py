"""
Test complete workflow of Generator and Discriminator
1. Use Generator to generate samples
2. Use Discriminator to evaluate sample quality
"""
import torch
from test_generator import load_generator, generate_sample
from test_discriminator import load_discriminator, evaluate_text

# --- Configuration ---
GENERATOR_PATH = "./llama3-synthetic-patient-generator"
DISCRIMINATOR_PATH = "./llama3-phi-discriminator"
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def main():
    """Main function"""
    print("=" * 60)
    print("Generator + Discriminator Test")
    print("=" * 60)
    
    # 1. Load Generator
    print("\n[1/3] Loading Generator...")
    generator, gen_tokenizer = load_generator(GENERATOR_PATH, BASE_MODEL_NAME)
    
    # 2. Load Discriminator
    print("\n[2/3] Loading Discriminator...")
    discriminator, disc_tokenizer = load_discriminator(DISCRIMINATOR_PATH, BASE_MODEL_NAME)
    
    # 3. Generate and evaluate
    print("\n[3/3] Generating and evaluating samples...")
    print("=" * 60)
    
    test_prompts = [
        "Generate discharge record for: Sex: Male, Medical Service: MEDICINE",
        "Generate discharge record for: Sex: Female, Medical Service: CARDIOLOGY",
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        # Generate
        print("Generating...")
        generated_text = generate_sample(generator, gen_tokenizer, prompt)
        print(f"Generated {len(generated_text)} characters")
        
        # Evaluate
        print("Evaluating...")
        scores = evaluate_text(discriminator, disc_tokenizer, generated_text)
        
        # Display results
        print("\nResults:")
        print(f"  Realism Score: {scores['realism']:.4f}")
        print(f"  Difficulty Score: {scores['difficulty']:.4f}")
        print(f"  Quality: {scores['quality']}")
        
        # Save results
        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "scores": scores
        }
        results.append(result)
        
        # Save to file
        output_file = f"test_result_{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write("Generated Text:\n")
            f.write(generated_text)
            f.write("\n\nEvaluation:\n")
            f.write(f"Realism: {scores['realism']:.4f}\n")
            f.write(f"Difficulty: {scores['difficulty']:.4f}\n")
            f.write(f"Quality: {scores['quality']}\n")
        print(f"Saved to {output_file}")
        print("-" * 60)
    
    # Statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    avg_realism = sum(r["scores"]["realism"] for r in results) / len(results)
    good_count = sum(1 for r in results if r["scores"]["quality"] == "good")
    
    print(f"Total samples: {len(results)}")
    print(f"Average realism: {avg_realism:.4f}")
    print(f"High quality samples (realism >= 0.8): {good_count}/{len(results)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
