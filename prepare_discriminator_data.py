"""
Prepare Discriminator training data
Convert CSV files to training format
"""
import csv
import json
import os

def prepare_discriminator_data(good_csv_path, bad_csv_path, output_path):
    """
    Prepare discriminator training data
    
    Args:
        good_csv_path: Path to good data CSV file
        bad_csv_path: Path to bad data CSV file
        output_path: Path to output JSONL file
    """
    data = []
    
    print("Loading good data...")
    with open(good_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 1000:  # Only take first 1000 rows
                break
            try:
                # Extract key fields
                text = row.get('filled_text', '')
                # Use realism score from CSV as label (follows preprocess.py scoring rules)
                realism = float(row.get('realism', 1.0)) if row.get('realism') else 1.0
                difficulty = float(row.get('DIFFICULTY', 0.0)) if row.get('DIFFICULTY') else 0.0
                # Extract overall field (contains detailed error information, important for Discriminator)
                overall = row.get('overall', '')
                # Extract dimension scores (for analysis and validation)
                linguistic_realism = float(row.get('linguistic_realism', 0.0)) if row.get('linguistic_realism') else 0.0
                clinical_realism = float(row.get('clinical_consistency_realism', 0.0)) if row.get('clinical_consistency_realism') else 0.0
                structure_realism = float(row.get('document_structure_realism', 0.0)) if row.get('document_structure_realism') else 0.0
                
                if text.strip():
                    data.append({
                        "text": text,
                        "label": realism,  # Use realism score as label, not simple 1.0
                        "realism": realism,
                        "difficulty": difficulty,
                        "overall": overall,  # Save detailed error information (for analysis and validation)
                        "linguistic_realism": linguistic_realism,
                        "clinical_realism": clinical_realism,
                        "structure_realism": structure_realism,
                    })
            except Exception as e:
                print(f"Error processing good data row {i}: {e}")
                continue
    
    print(f"Loaded {len(data)} good samples")
    
    print("Loading bad data...")
    bad_count = 0
    with open(bad_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if bad_count >= 1000:  # Only take first 1000 rows
                break
            try:
                text = row.get('filled_text', '')
                # Use realism score from CSV as label (follows preprocess.py scoring rules)
                realism = float(row.get('realism', 0.0)) if row.get('realism') else 0.0
                difficulty = float(row.get('DIFFICULTY', 0.0)) if row.get('DIFFICULTY') else 0.0
                # Extract overall field (contains detailed error information, important for Discriminator)
                overall = row.get('overall', '')
                # Extract dimension scores (for analysis and validation)
                linguistic_realism = float(row.get('linguistic_realism', 0.0)) if row.get('linguistic_realism') else 0.0
                clinical_realism = float(row.get('clinical_consistency_realism', 0.0)) if row.get('clinical_consistency_realism') else 0.0
                structure_realism = float(row.get('document_structure_realism', 0.0)) if row.get('document_structure_realism') else 0.0
                
                if text.strip():
                    data.append({
                        "text": text,
                        "label": realism,  # Use realism score as label, not simple 0.0
                        "realism": realism,
                        "difficulty": difficulty,
                        "overall": overall,  # Save detailed error information (for analysis and validation)
                        "linguistic_realism": linguistic_realism,
                        "clinical_realism": clinical_realism,
                        "structure_realism": structure_realism,
                    })
                    bad_count += 1
            except Exception as e:
                print(f"Error processing bad data row {i}: {e}")
                continue
    
    print(f"Loaded {bad_count} bad samples")
    print(f"Total samples: {len(data)}")
    
    # Statistics on realism score distribution (validate data quality)
    if data:
        realism_scores = [item['realism'] for item in data]
        good_realism = [item['realism'] for item in data if item['label'] >= 0.7]  # Assume good data realism >= 0.7
        bad_realism = [item['realism'] for item in data if item['label'] < 0.7]
        
        print(f"\nRealism score statistics:")
        print(f"  All samples: min={min(realism_scores):.4f}, max={max(realism_scores):.4f}, mean={sum(realism_scores)/len(realism_scores):.4f}")
        if good_realism:
            print(f"  Good samples (realism >= 0.7): min={min(good_realism):.4f}, max={max(good_realism):.4f}, mean={sum(good_realism)/len(good_realism):.4f}, count={len(good_realism)}")
        if bad_realism:
            print(f"  Bad samples (realism < 0.7): min={min(bad_realism):.4f}, max={max(bad_realism):.4f}, mean={sum(bad_realism)/len(bad_realism):.4f}, count={len(bad_realism)}")
        
        # Validation: good data realism should generally be higher than bad data
        if good_realism and bad_realism:
            avg_good = sum(good_realism) / len(good_realism)
            avg_bad = sum(bad_realism) / len(bad_realism)
            if avg_good > avg_bad:
                print(f"  ✓ Validation passed: Good data realism ({avg_good:.4f}) > Bad data realism ({avg_bad:.4f})")
            else:
                print(f"  ⚠ Warning: Good data realism ({avg_good:.4f}) <= Bad data realism ({avg_bad:.4f})")
                print(f"    This may indicate data quality issues. Please check the CSV files.")
        
        # Statistics on overall field coverage
        overall_count = sum(1 for item in data if item.get('overall', '').strip())
        print(f"\nOverall field statistics:")
        print(f"  Samples with overall info: {overall_count}/{len(data)} ({100*overall_count/len(data):.1f}%)")
        
        # Analyze main error types (extracted from overall field)
        if overall_count > 0:
            error_types = {"LINGUISTIC": 0, "CLINICAL": 0, "STRUCTURE": 0}
            for item in data:
                overall_text = item.get('overall', '')
                if 'LINGUISTIC' in overall_text.upper():
                    error_types["LINGUISTIC"] += 1
                if 'CLINICAL' in overall_text.upper():
                    error_types["CLINICAL"] += 1
                if 'STRUCTURE' in overall_text.upper() or 'STRUCT' in overall_text.upper():
                    error_types["STRUCTURE"] += 1
            
            print(f"  Error type distribution:")
            for err_type, count in error_types.items():
                print(f"    {err_type}: {count} samples")
    
    # Save as JSONL format
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} samples to {output_path}")
    print(f"\nNote: Using realism scores from preprocess.py as labels (not simple 1.0/0.0)")
    print(f"This ensures the Discriminator learns the same scoring rules as preprocess.py")
    print(f"\nAdditional metadata saved:")
    print(f"  - overall: Detailed error information (e.g., 'Main deduction:LINGUISTIC(0.45) | ...')")
    print(f"  - linguistic_realism, clinical_realism, structure_realism: Individual dimension scores")
    print(f"  These fields are saved for analysis and validation, but not used directly in training.")
    return data

if __name__ == "__main__":
    import sys
    
    # Path configuration - try multiple possible locations
    possible_paths = [
        ("preprocess_data_label_good_phi.csv", "preprocess_data_label_bad_phi.csv"),
        ("../preprocess_data_label_good_phi.csv", "../preprocess_data_label_bad_phi.csv"),
        ("../generate_data/generate_and_preprocess/preprocess_data_label_good_phi.csv", 
         "../generate_data/generate_and_preprocess/preprocess_data_label_bad_phi.csv"),
    ]
    
    good_csv = None
    bad_csv = None
    
    for good_path, bad_path in possible_paths:
        if os.path.exists(good_path) and os.path.exists(bad_path):
            good_csv = good_path
            bad_csv = bad_path
            print(f"Found CSV files:")
            print(f"  Good data: {good_csv}")
            print(f"  Bad data: {bad_csv}")
            break
    
    if good_csv is None or bad_csv is None:
        print("Error: Could not find CSV files!")
        print("Please ensure the following files exist:")
        print("  - preprocess_data_label_good_phi.csv")
        print("  - preprocess_data_label_bad_phi.csv")
        print("\nTried paths:")
        for good_path, bad_path in possible_paths:
            print(f"  - {good_path} (exists: {os.path.exists(good_path)})")
            print(f"  - {bad_path} (exists: {os.path.exists(bad_path)})")
        sys.exit(1)
    
    output_jsonl = "discriminator_train.jsonl"
    
    # Prepare data
    data = prepare_discriminator_data(good_csv, bad_csv, output_jsonl)
    print(f"\nData preparation complete! Total samples: {len(data)}")



