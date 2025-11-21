#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare good data and bad data examples

Used to verify that fill_phi_bad_data.py correctly generated low-quality data
"""

import csv
import json
import sys

def analyze_single_record(row, label=""):
    """Analyze quality of a single record"""
    filled_text = row.get('filled_text', '')
    phi_annotations = row.get('phi_annotations', '[]')
    bad_type = row.get('bad_data_type', 'N/A')
    
    print(f"\n{'='*80}")
    print(f"{label}")
    if bad_type != 'N/A':
        print(f"Bad Data Type: {bad_type}")
    print(f"{'='*80}")
    
    # Show first 500 characters
    print("\n[Filled Text Preview]:")
    print(filled_text[:500])
    if len(filled_text) > 500:
        print("...")
    
    # Parse and display PHI annotations
    try:
        annotations = json.loads(phi_annotations)
        print(f"\n[PHI Annotations] (Total: {len(annotations)}):")
        for i, ann in enumerate(annotations[:10], 1):  # 只显示前10个
            print(f"  {i}. {ann.get('type', 'UNKNOWN'):30s} = {ann.get('value', '')}")
        if len(annotations) > 10:
            print(f"  ... and {len(annotations) - 10} more")
    except:
        print(f"\n[PHI Annotations]: Failed to parse")
    
    # Detect potential issues
    print(f"\n[Quality Check]:")
    
    issues = []
    
    # Check name
    if 'Name:' in filled_text:
        import re
        name_match = re.search(r'Name:\s+(.+?)(?:\s+Unit No:|$)', filled_text)
        if name_match:
            name = name_match.group(1).strip()
            if len(name.split()) < 2:
                issues.append(f"⚠️  Name has only 1 word: '{name}'")
            if not name[0].isupper() if name else False:
                issues.append(f"⚠️  Name not capitalized: '{name}'")
            if any(c in name for c in '@#$123'):
                issues.append(f"⚠️  Name contains weird characters: '{name}'")
            if len(name) == 1:
                issues.append(f"⚠️  Name is single letter: '{name}'")
    
    # Check dates
    if 'Admission Date:' in filled_text and 'Discharge Date:' in filled_text:
        import re
        from datetime import datetime
        adm_match = re.search(r'Admission Date:\s+(\d{4}-\d{2}-\d{2})', filled_text)
        dis_match = re.search(r'Discharge Date:\s+(\d{4}-\d{2}-\d{2})', filled_text)
        if adm_match and dis_match:
            try:
                adm_date = datetime.strptime(adm_match.group(1), '%Y-%m-%d')
                dis_date = datetime.strptime(dis_match.group(1), '%Y-%m-%d')
                if adm_date >= dis_date:
                    issues.append(f"⚠️  Admission >= Discharge: {adm_match.group(1)} to {dis_match.group(1)}")
                if dis_date > datetime.now():
                    issues.append(f"⚠️  Future date: {dis_match.group(1)}")
                if adm_date.year < 1920:
                    issues.append(f"⚠️  Too old date: {adm_match.group(1)}")
            except:
                pass
    
    # Check Attending
    if 'Attending:' not in filled_text:
        issues.append("⚠️  Missing 'Attending:' field")
    else:
        import re
        att_match = re.search(r'Attending:\s+(.+)', filled_text)
        if att_match:
            attending = att_match.group(1).strip()
            if not attending.startswith('Dr.'):
                issues.append(f"⚠️  Attending not starting with 'Dr.': '{attending}'")
    
    # Check keywords
    keywords = [
        'Discharge Disposition:',
        'Discharge Diagnosis:',
        'Discharge Condition:',
        'Discharge Instructions:',
        'Followup Instructions:'
    ]
    missing_keywords = [kw for kw in keywords if kw not in filled_text]
    if missing_keywords:
        issues.append(f"⚠️  Missing keywords: {', '.join(missing_keywords)}")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅ No obvious issues detected")
    
    print(f"{'='*80}\n")


def compare_files(good_file, bad_file, num_samples=3):
    """Compare good data and bad data files"""
    print("="*80)
    print("GOOD vs BAD DATA COMPARISON")
    print("="*80)
    
    # Read good data
    print(f"\nReading good data from: {good_file}")
    try:
        with open(good_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            good_rows = [row for row in reader if row.get('filled_text', '').strip()]
    except FileNotFoundError:
        print(f"❌ File not found: {good_file}")
        return
    
    # Read bad data
    print(f"Reading bad data from: {bad_file}")
    try:
        with open(bad_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            bad_rows = [row for row in reader if row.get('filled_text', '').strip()]
    except FileNotFoundError:
        print(f"❌ File not found: {bad_file}")
        return
    
    print(f"\nGood data records: {len(good_rows)}")
    print(f"Bad data records: {len(bad_rows)}")
    
    # Compare samples
    import random
    random.seed(42)
    
    if len(good_rows) >= num_samples:
        good_samples = random.sample(good_rows, num_samples)
    else:
        good_samples = good_rows
    
    if len(bad_rows) >= num_samples:
        bad_samples = random.sample(bad_rows, num_samples)
    else:
        bad_samples = bad_rows
    
    # Analyze good data
    print("\n" + "🟢"*40)
    print("GOOD DATA SAMPLES")
    print("🟢"*40)
    for i, row in enumerate(good_samples, 1):
        analyze_single_record(row, f"Good Sample #{i}")
    
    # Analyze bad data
    print("\n" + "🔴"*40)
    print("BAD DATA SAMPLES")
    print("🔴"*40)
    for i, row in enumerate(bad_samples, 1):
        analyze_single_record(row, f"Bad Sample #{i}")
    
    # Statistics of bad data types
    if bad_rows and 'bad_data_type' in bad_rows[0]:
        print("\n" + "="*80)
        print("BAD DATA TYPE DISTRIBUTION")
        print("="*80)
        from collections import Counter
        bad_types = [row['bad_data_type'] for row in bad_rows if row.get('bad_data_type')]
        type_counts = Counter(bad_types)
        for bad_type, count in sorted(type_counts.items()):
            percentage = count / len(bad_rows) * 100
            print(f"  {bad_type:30s}: {count:5d} ({percentage:5.1f}%)")


if __name__ == "__main__":
    # Default file paths
    good_file = "patient_extracted_with_synth_phi.csv"
    bad_file = "patient_extracted_with_bad_phi.csv"
    
    # Can be specified from command line
    if len(sys.argv) >= 3:
        good_file = sys.argv[1]
        bad_file = sys.argv[2]
    
    num_samples = 3
    if len(sys.argv) >= 4:
        try:
            num_samples = int(sys.argv[3])
        except:
            pass
    
    compare_files(good_file, bad_file, num_samples)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nUsage: python3 {sys.argv[0]} [good_file] [bad_file] [num_samples]")

