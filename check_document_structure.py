#!/usr/bin/env python3
"""
检查 patient_notes_clinical_consistency_scored.csv 中的文档格式
并计算 document structure score
"""

import csv
import re
import sys

def get_non_empty_lines(text):
    """获取非空行"""
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

def check_structure_order(text):
    """
    检查前3行（非空格）的格式
    返回: (score, details)
    - 第一行必须是 "Name: ... Unit No: ..."（顺序不能变）
    - 第二行必须是 "Admission Date: ... Discharge Date: ..."
    - 第三行必须是 "Date of Birth: ... Sex: ..."
    """
    non_empty_lines = get_non_empty_lines(text)
    
    if len(non_empty_lines) < 3:
        return 0.0, "Less than 3 non-empty lines"
    
    score = 0.0
    details = []
    
    # 检查第一行: Name: ... Unit No: ...
    first_line = non_empty_lines[0]
    if re.search(r'Name:\s+.*Unit No:', first_line, re.IGNORECASE):
        score += 1.0 / 3.0
        details.append("Line 1 format correct")
    else:
        details.append("Line 1 format error: missing 'Name: ... Unit No:' pattern")
    
    # 检查第二行: Admission Date: ... Discharge Date: ...
    second_line = non_empty_lines[1]
    if re.search(r'Admission Date:\s+.*Discharge Date:', second_line, re.IGNORECASE):
        score += 1.0 / 3.0
        details.append("Line 2 format correct")
    else:
        details.append("Line 2 format error: missing 'Admission Date: ... Discharge Date:' pattern")
    
    # 检查第三行: Date of Birth: ... Sex: ...
    third_line = non_empty_lines[2]
    if re.search(r'Date of Birth:\s+.*Sex:', third_line, re.IGNORECASE):
        score += 1.0 / 3.0
        details.append("Line 3 format correct")
    else:
        details.append("Line 3 format error: missing 'Date of Birth: ... Sex:' pattern")
    
    return score, "; ".join(details)

def check_attending(text):
    """检查是否包含 Attending: (大写)"""
    if re.search(r'Attending:\s+', text):
        return 1.0, "Attending present"
    return 0.0, "Attending missing"

def check_keywords(text):
    """
    检查是否包含关键词
    - Discharge Disposition:
    - Discharge Diagnosis:
    - Discharge Condition:
    - Discharge Instructions:
    - Followup Instructions:
    """
    keywords = [
        'Discharge Disposition:',
        'Discharge Diagnosis:',
        'Discharge Condition:',
        'Discharge Instructions:',
        'Followup Instructions:'
    ]
    
    found_count = 0
    found_keywords = []
    missing_keywords = []
    
    for keyword in keywords:
        if re.search(re.escape(keyword), text, re.IGNORECASE):
            found_count += 1
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    score = found_count / len(keywords)
    details = f"Found {found_count}/{len(keywords)} keywords"
    if found_keywords:
        details += f" [Found: {', '.join(found_keywords)}]"
    if missing_keywords:
        details += f" [Missing: {', '.join(missing_keywords)}]"
    
    return score, details

def calculate_document_structure_score(text):
    """
    计算 document structure score
    - 顺序检查占70% (前3行格式 + Attending)
    - 关键词检查占30%
    返回: (score, reason)
    """
    # 顺序检查 (70%)
    order_score, order_details = check_structure_order(text)
    attending_score, attending_details = check_attending(text)
    
    # 前3行格式占顺序检查的75%，Attending占25%
    order_total = order_score * 0.75 + attending_score * 0.25
    
    # 关键词检查 (30%)
    keyword_score, keyword_details = check_keywords(text)
    
    # 最终分数
    final_score = order_total * 0.7 + keyword_score * 0.3
    
    # 构建详细原因
    reason_parts = []
    reason_parts.append(f"Order check (70%): First 3 lines format ({order_score:.2f})*0.75 + Attending ({attending_score:.2f})*0.25 = {order_total:.2f}")
    reason_parts.append(f"  - {order_details}")
    reason_parts.append(f"  - {attending_details}")
    reason_parts.append(f"Keyword check (30%): {keyword_score:.2f}")
    reason_parts.append(f"  - {keyword_details}")
    reason_parts.append(f"Total score: {order_total:.4f}*0.7 + {keyword_score:.4f}*0.3 = {final_score:.4f}")
    
    reason = " | ".join(reason_parts)
    
    return final_score, reason

def process_csv(input_file, output_file):
    """处理CSV文件，添加 document_structure_score 列"""
    rows = []
    
    # 读取CSV文件
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # 移除已存在的列（如果存在）
        if 'document_structure_score' in fieldnames:
            print("Warning: document_structure_score column already exists, will be overwritten")
            fieldnames = [f for f in fieldnames if f != 'document_structure_score']
        if 'document_structure_score_reason' in fieldnames:
            print("Warning: document_structure_score_reason column already exists, will be overwritten")
            fieldnames = [f for f in fieldnames if f != 'document_structure_score_reason']
        
        fieldnames.append('document_structure_score')
        fieldnames.append('document_structure_score_reason')
        
        for i, row in enumerate(reader):
            extracted_text = row.get('extracted_text', '')
            
            # 计算分数和原因
            score, reason = calculate_document_structure_score(extracted_text)
            row['document_structure_score'] = f"{score:.4f}"
            row['document_structure_score_reason'] = reason
            
            rows.append(row)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} rows...")
    
    # 写入CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Processing complete! Total rows processed: {len(rows)}")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    input_file = 'patient_notes_clinical_consistency_scored.csv'
    output_file = 'patient_notes_clinical_consistency_scored.csv'
    
    print("Starting document structure check...")
    process_csv(input_file, output_file)

