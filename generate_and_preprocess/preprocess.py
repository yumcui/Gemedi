#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified preprocessing script - directly generates preprocess_data_label.csv

Integrates logic from three modules:
1. Document Structure Realism (from check_document_structure.py)
2. Linguistic Realism & Difficulty (from editor.py)
3. Clinical Consistency (from score_clinical_consistency.py)

Final output columns:
- extracted_text, filled_text, phi_annotations
- score_age (from LLM, divided by 10, evaluates age-diagnosis relevance)
- score_sex (from LLM, divided by 10, evaluates sex-diagnosis relevance)
- clinical_consistency_realism (weighted average based on score_age and score_sex)
- document_structure_realism, document_structure_score_reason
- linguistic_realism, phi_amount_score, ambiguity_score, DIFFICULTY
- realism (calculated value)
"""

import csv
import json
import re
import sys
import time
import math
import ollama
from datetime import datetime, date
from dateutil import parser
from enum import Enum
from typing import Optional, Tuple
import concurrent.futures

# ============================================================
# Configuration Parameters
# ============================================================
INPUT_CSV = "patient_extracted_with_bad_phi.csv"
OUTPUT_CSV = "preprocess_data_label_bad_phi.csv"

# Clinical Consistency Configuration
MODEL_NAME = "llama3.1"  # LLM model name
MAX_WORKERS = 15  # Number of concurrent threads
AGE_WEIGHT = 0.5  # Age weight
SEX_WEIGHT = 0.5  # Sex weight
USE_FULL_TEXT = True

# Realism Weight Configuration
LINGUISTIC_WEIGHT = 0.45
CLINICAL_WEIGHT = 0.30
STRUCTURE_WEIGHT = 0.25


# ============================================================
# MODULE 1: Document Structure Realism
# (from check_document_structure.py)
# ============================================================

def get_non_empty_lines(text):
    """Get non-empty lines"""
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]


def check_structure_order(text):
    """
    Check format of first 3 lines (non-empty)
    Returns: (score, details)
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
        details.append("Line 1 format error")
    
    # 检查第二行: Admission Date: ... Discharge Date: ...
    second_line = non_empty_lines[1]
    if re.search(r'Admission Date:\s+.*Discharge Date:', second_line, re.IGNORECASE):
        score += 1.0 / 3.0
        details.append("Line 2 format correct")
    else:
        details.append("Line 2 format error")
    
    # 检查第三行: Date of Birth: ... Sex: ...
    third_line = non_empty_lines[2]
    if re.search(r'Date of Birth:\s+.*Sex:', third_line, re.IGNORECASE):
        score += 1.0 / 3.0
        details.append("Line 3 format correct")
    else:
        details.append("Line 3 format error")
    
    return score, "; ".join(details)


def check_attending(text):
    """Check if contains Attending:"""
    if re.search(r'Attending:\s+', text):
        return 1.0, "Attending present"
    return 0.0, "Attending missing"


def check_keywords(text):
    """Check if contains keywords"""
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
    Calculate document structure score
    Returns: (score, reason)
    """
    order_score, order_details = check_structure_order(text)
    attending_score, attending_details = check_attending(text)
    order_total = order_score * 0.75 + attending_score * 0.25
    
    keyword_score, keyword_details = check_keywords(text)
    final_score = order_total * 0.7 + keyword_score * 0.3
    
    reason_parts = []
    reason_parts.append(f"Order check (70%): First 3 lines format ({order_score:.2f})*0.75 + Attending ({attending_score:.2f})*0.25 = {order_total:.2f}")
    reason_parts.append(f"  - {order_details}")
    reason_parts.append(f"  - {attending_details}")
    reason_parts.append(f"Keyword check (30%): {keyword_score:.2f}")
    reason_parts.append(f"  - {keyword_details}")
    reason_parts.append(f"Total score: {order_total:.4f}*0.7 + {keyword_score:.4f}*0.3 = {final_score:.4f}")
    
    reason = " | ".join(reason_parts)
    return final_score, reason


# ============================================================
# MODULE 2: Linguistic Realism & Difficulty
# (from editor.py)
# ============================================================

def has_weird_char(s: str) -> bool:
    """Check if contains weird characters
    Allowed characters: letters, numbers, spaces, common punctuation (. - / , : ; ( ) ')
    Not allowed: special Unicode characters, control characters, etc.
    """
    if not isinstance(s, str):
        return True
    # 允许常见标点符号：. - / , : ; ( ) '
    return re.search(r"[^A-Za-z0-9\s\.\-\/,:;\(\)']", s) is not None


def parse_date(value: str):
    """Try to parse string as date object"""
    if not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except Exception:
            continue
    return None


def is_name_type(t: str) -> bool:
    """Check if PHI type is name type"""
    if not isinstance(t, str):
        return False
    u = t.upper()
    return ("NAME" in u) and ("TITLE" not in u)


def is_doctor_type(t: str) -> bool:
    """Check if PHI type is doctor name type"""
    if not isinstance(t, str):
        return False
    u = t.upper()
    return ("PROVIDER" in u) or ("DOCTOR" in u) or ("PHYSICIAN" in u)


def is_date_type(t: str) -> bool:
    """Check if PHI type is date type"""
    if not isinstance(t, str):
        return False
    u = t.upper()
    return ("DATE" in u) or ("DOB" in u) or ("BIRTH" in u)


def get_admission_type_match(t: str) -> bool:
    """Check if is admission date type"""
    if not isinstance(t, str):
        return False
    u = t.upper()
    return ("ADMISSION" in u) and ("DATE" in u)


def get_discharge_type_match(t: str) -> bool:
    """Check if is discharge date type"""
    if not isinstance(t, str):
        return False
    u = t.upper()
    return ("DISCHARGE" in u) and ("DATE" in u)


def is_capitalized_name_word(word: str) -> bool:
    """Check if word is capitalized"""
    if not word:
        return False
    if not word[0].isalpha():
        return True
    return word[0].isupper()


def name_all_words_capitalized(name: str) -> bool:
    """Check if all letter-containing words in name are capitalized"""
    if not isinstance(name, str):
        return False
    words = name.strip().split()
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if not alpha_words:
        return False
    return all(is_capitalized_name_word(w) for w in alpha_words)


def compute_linguistic_for_row(phi_annotations):
    """
    Calculate linguistic score (0~1)
    Based on 8 rules + improved penalty mechanism
    Returns: (score, details_list) where details_list records specific rule violations
    """
    if not isinstance(phi_annotations, list):
        return (0.0, ["无PHI数据"])

    name_values = []  # 存储 (value, type) 元组，以便区分不同类型的姓名
    doctor_values = []
    date_values = []
    admission_dates = []
    discharge_dates = []
    all_phi_values = []  # 用于检查所有PHI字段的特殊字符

    for ann in phi_annotations:
        val = str(ann.get("value", "")).strip()
        typ = ann.get("type", "")

        if is_name_type(typ):
            name_values.append((val, typ))  # 存储值和类型
        if is_doctor_type(typ):
            doctor_values.append(val)
        if is_date_type(typ):
            date_values.append(val)
        if get_admission_type_match(typ):
            admission_dates.append(val)
        if get_discharge_type_match(typ):
            discharge_dates.append(val)
        
        # 收集所有PHI值用于特殊字符检查
        if val:
            all_phi_values.append((typ, val))

    rules_total = 0
    rules_passed = 0
    severe_violations = 0  # 严重违反计数
    violation_details = []  # 记录具体违反的规则

    # Rule 1: Name >= 2 words, and no weird characters
    # Note: DISCHARGE_INSTRUCTIONS_NAME (surname in salutation) can be a single word, this is normal
    if name_values:
        rules_total += 1
        ok = True
        problematic_names = []
        for val, typ in name_values:
            # Only check NAME type (patient full name), don't check DISCHARGE_INSTRUCTIONS_NAME (surname in salutation)
            if typ == "DISCHARGE_INSTRUCTIONS_NAME":
                # DISCHARGE_INSTRUCTIONS_NAME (surname in salutation) only checks special characters, not word count
                if has_weird_char(val):
                    ok = False
                    weird_chars = []
                    for char in val:
                        if re.search(r"[^A-Za-z0-9\s\.\-\/,:;\(\)']", char):
                            weird_chars.append(f"'{char}'(U+{ord(char):04X})")
                    if weird_chars:
                        problematic_names.append(f"称呼姓氏'{val[:30]}{'...' if len(val) > 30 else ''}'[特殊字符:{','.join(weird_chars[:3])}]")
                    severe_violations += 1  # 严重违反
            else:
                # NAME type (patient full name) needs >= 2 words
                if len(val.split()) < 2 or has_weird_char(val):
                    ok = False
                    if has_weird_char(val):
                        # 找出具体是哪个特殊字符
                        weird_chars = []
                        for char in val:
                            if re.search(r"[^A-Za-z0-9\s\.\-\/,:;\(\)']", char):
                                weird_chars.append(f"'{char}'(U+{ord(char):04X})")
                        if weird_chars:
                            problematic_names.append(f"'{val[:30]}{'...' if len(val) > 30 else ''}'[特殊字符:{','.join(weird_chars[:3])}]")
                        else:
                            problematic_names.append(f"'{val[:30]}{'...' if len(val) > 30 else ''}'[含特殊字符]")
                        severe_violations += 1  # 严重违反
                    elif len(val.strip()) == 1:
                        problematic_names.append(f"'{val}'[单个字母]")
                        severe_violations += 1  # 严重违反
                    else:
                        problematic_names.append(f"'{val[:30]}{'...' if len(val) > 30 else ''}'[少于2个单词]")
        
        if not ok and problematic_names:
            violation_details.append(f"姓名问题:{';'.join(problematic_names[:3])}")
        if ok:
            rules_passed += 1
    
    # Additional check: check if all PHI fields have special characters (for debugging)
    # Note: only check non-text fields (e.g., NAME, UNIT_NO, DATE, etc.), don't check text fields like FOLLOWUP_INSTRUCTIONS
    if all_phi_values:
        text_field_types = ["FOLLOWUP_INSTRUCTIONS", "DISCHARGE_INSTRUCTIONS"]  # 文本字段允许标点符号
        for typ, val in all_phi_values:
            # 跳过文本字段的特殊字符检查（这些字段允许标点符号）
            if any(text_type in typ.upper() for text_type in text_field_types):
                continue
            if has_weird_char(val):
                weird_chars = []
                for char in val:
                    if re.search(r"[^A-Za-z0-9\s\.\-\/,:;\(\)']", char):
                        weird_chars.append(f"'{char}'(U+{ord(char):04X})")
                if weird_chars and f"{typ}含特殊字符" not in "|".join(violation_details):
                    violation_details.append(f"{typ}含特殊字符:{val[:20]}{'...' if len(val) > 20 else ''}[{','.join(weird_chars[:2])}]")

    # Rule 2: All dates can be parsed
    if date_values:
        rules_total += 1
        ok = True
        for v in date_values:
            if parse_date(v) is None:
                ok = False
                violation_details.append("日期无法解析")
                severe_violations += 1  # 无法解析日期是严重问题
                break
        if ok:
            rules_passed += 1

    # Rule 3: Doctor name starts with Dr / Dr.
    if doctor_values:
        rules_total += 1
        ok = True
        for v in doctor_values:
            s = v.lower()
            if not (s.startswith("dr ") or s.startswith("dr.") or s == "dr"):
                ok = False
                violation_details.append("医生姓名格式错误(缺少Dr/Dr.)")
                # 医生姓名格式错误是中等严重问题
                break
        if ok:
            rules_passed += 1

    # Rule 4: Name cannot be a single letter
    if name_values:
        rules_total += 1
        ok = True
        for v, typ in name_values:
            # DISCHARGE_INSTRUCTIONS_NAME (surname in salutation) can be a single letter, skip check
            if typ == "DISCHARGE_INSTRUCTIONS_NAME":
                continue
            if len(v.strip()) == 1:
                ok = False
                if "姓名是单个字母" not in violation_details:  # 避免重复
                    violation_details.append("姓名是单个字母")
                severe_violations += 1  # 严重违反
                break
        if ok:
            rules_passed += 1

    # Rule 5: All words in name are capitalized
    if name_values:
        rules_total += 1
        ok = True
        for v, typ in name_values:
            # DISCHARGE_INSTRUCTIONS_NAME (surname in salutation) can be a single word, skip check
            if typ == "DISCHARGE_INSTRUCTIONS_NAME":
                continue
            if not name_all_words_capitalized(v):
                ok = False
                violation_details.append("姓名首字母未大写")
                break
        if ok:
            rules_passed += 1

    # Rule 6: Date cannot be in the future
    if date_values:
        rules_total += 1
        today = date.today()
        ok = True
        for v in date_values:
            d = parse_date(v)
            if d is None or d > today:
                ok = False
                if d and d > today:
                    violation_details.append("日期是未来日期")
                    severe_violations += 1  # 未来日期是严重问题
                break
        if ok:
            rules_passed += 1

    # Rule 7: Date cannot be earlier than 105 years ago
    if date_values:
        rules_total += 1
        today = date.today()
        if today.year >= 105:
            cutoff = today.replace(year=today.year - 105)
        else:
            cutoff = date(1900, 1, 1)
        ok = True
        for v in date_values:
            d = parse_date(v)
            if d is None or d < cutoff:
                ok = False
                if d and d < cutoff:
                    violation_details.append("日期过旧(>105年前)")
                    severe_violations += 1  # 太旧的日期是严重问题
                break
        if ok:
            rules_passed += 1

    # Rule 8: Admission Date < Discharge Date
    if admission_dates and discharge_dates:
        rules_total += 1
        adm = parse_date(admission_dates[0])
        dis = parse_date(discharge_dates[0])
        if adm is not None and dis is not None and adm < dis:
            rules_passed += 1
        elif adm and dis and adm >= dis:
            violation_details.append("入院日期>=出院日期")
            severe_violations += 1  # 入院晚于出院是严重逻辑错误

    if rules_total == 0:
        return (0.0, ["无规则可检查"])
    
    # Improved scoring: use stricter penalty mechanism
    base_score = rules_passed / rules_total
    violations = rules_total - rules_passed
    
    if violations > 0:
        # If there are severe violations (special characters, single letter, future date, etc.), directly heavily penalize
        if severe_violations > 0:
            # Severe violations: directly significantly reduce score, regardless of original score
            # Example: original 1.0, 1 severe violation → directly reduce to 0.5
            # Example: original 1.0, 2 severe violations → directly reduce to 0.3
            severe_penalty_ratio = severe_violations / rules_total
            
            if severe_penalty_ratio >= 0.5:
                # Severe violation ratio >= 50%, directly reduce to very low
                final_score = 0.2
            elif severe_penalty_ratio >= 0.3:
                # Severe violation ratio >= 30%, significantly reduce
                final_score = 0.3
            elif severe_penalty_ratio >= 0.2:
                # Severe violation ratio >= 20%, moderately reduce
                final_score = 0.4
            else:
                # Severe violation ratio < 20%, but still heavily penalize
                final_score = 0.5
            
            # Ensure it doesn't exceed base score (even with severe violations, can't score too high because other rules passed)
            final_score = min(final_score, base_score * 0.6)
        else:
            # Only general violations, use moderate penalty
            penalty_factor = (violations / rules_total) ** 1.5
            penalty_strength = 0.5
            final_score = base_score * (1 - penalty_factor * penalty_strength)
    else:
        final_score = base_score
        violation_details = ["No violations"]  # If no violations, record as no violations
    
    # Ensure score is in [0, 1] range
    final_score = max(0.0, min(1.0, final_score))
    
    # Return score and detailed information
    return (final_score, violation_details if violation_details else ["No violations"])


# Ambiguous words list
AMBIGUOUS_WORDS = sorted(set([
    "smith","johnson","williams","brown","jones","garcia","miller","davis",
    "rodriguez","martinez","hernandez","lopez","gonzalez","wilson","anderson",
    "thomas","taylor","moore","jackson","martin","lee","perez","thompson",
    "white","harris","sanchez","clark","ramirez","lewis","robinson","walker",
    "young","allen","wright","scott","torres","nguyen","hill","flores",
    "green","adams","nelson","baker","hall","rivera","campbell","mitchell",
    "carter","roberts","gomez","phillips","evans","turner","diaz","parker",
    "cruz","edwards","collins","reyes","stewart","morris","morales","murphy",
    "cook","rogers","gutierrez","ortiz","morgan","cooper","peterson","bailey",
    "reed","kelly","howard","ramos","kim","cox","ward","richardson","watson",
    "brooks","sandoval","price","bennett","wood","barnes","ross","butler",
    "powell","long","hughes","foster","gonzales",
    "general","memorial","mercy","grace","hope","saint","st","st.","community",
    "regional","county","medical","clinic","hospital","center","health",
    "children","women","university","city","district","foundation","trust",
    "washington","madison","lincoln","clinton","georgia","virginia","carolina",
    "kentucky","augusta","salem","florence","athens","paris","london",
    "oxford","cambridge","windsor","hamilton","victoria","dublin","sydney",
    "melbourne","providence","richmond","savannah","orlando","phoenix",
    "boston","denver","austin","houston","dallas","charlotte","charleston",
    "rochester","albany","baltimore","seattle","tampa","miami","atlanta",
    "kansas","tucson","columbus","indianapolis","minneapolis","pittsburgh",
    "valley","ridge","bay","park","forest","grove","heights","meadows","vista",
    "river","lake","island","point","harbor","harbour","plaza","square",
    "mount","mountain","hill","spring","springs","field","fields","bridge",
    "view","village","plains","union","central","station","house","hall",
    "may","will","summer","autumn","winter","march","april","june","july",
    "august","rose","ivy","dawn","angel","faith","joy","charity","light",
    "gold","queen","black","silver",
    "system","services","care","group","partners","network","alliance",
    "institute","center","centers","associates","solutions","consultants"
]))


def compute_phi_amount_score(n_phi: int) -> float:
    """Calculate phi_amount_score based on PHI count"""
    if n_phi <= 0:
        return 0.0
    if n_phi <= 12:
        return (n_phi / 12.0) * 0.8
    if n_phi < 16:
        return 0.8 + (n_phi - 12) / 4.0 * 0.2
    return 1.0


def compute_ambiguity_score(phi_annotations) -> float:
    """Calculate ambiguity_score"""
    if not isinstance(phi_annotations, list) or len(phi_annotations) == 0:
        return 0.0
    n = len(phi_annotations)
    amb_count = 0

    for ann in phi_annotations:
        val = str(ann.get("value", "")).lower()
        if any(word in val for word in AMBIGUOUS_WORDS):
            amb_count += 1

    return amb_count / n


def compute_difficulty_for_row(phi_annotations):
    """
    计算 phi_amount_score, ambiguity_score, DIFFICULTY
    DIFFICULTY = 0.7 * phi_amount_score + 0.3 * ambiguity_score
    """
    if not isinstance(phi_annotations, list):
        phi_annotations = []

    n_phi = len(phi_annotations)
    phi_amount_score = compute_phi_amount_score(n_phi)
    ambiguity_score = compute_ambiguity_score(phi_annotations)
    difficulty = 0.7 * phi_amount_score + 0.3 * ambiguity_score

    return phi_amount_score, ambiguity_score, difficulty


# ============================================================
# MODULE 3: Clinical Consistency
# (from score_clinical_consistency.py)
# ============================================================

class ConsistencyCheck(Enum):
    AGE = "age"
    SEX = "biological sex"


class ClinicalConsistencyEvaluator:
    @staticmethod
    def get_relevance(check_type: ConsistencyCheck, value: str, diagnosis: str, 
                     clinical_text: str, retries: int = 2) -> int:
        """使用 LLM 评估临床一致性"""
        if check_type == ConsistencyCheck.AGE:
            criteria = (
                "epidemiological likelihood and age-diagnosis relevance. Evaluate how RELEVANT and CONSISTENT "
                "the Primary Diagnosis is with a patient of this age. Higher score = more relevant/consistent."
            )
            if value == "-1":
                return -1
        elif check_type == ConsistencyCheck.SEX:
            criteria = (
                "anatomical compatibility and biological sex predilection. "
                "Check if the Primary Diagnosis is consistent with the patient's biological sex. "
                "Some conditions are more common in one sex, while others are sex-specific."
            )
            if value == None:
                return -1
            
            # 添加性别相关的上下文 - 改进版，更强调相关性评分
            sex_context = ""
            if value.upper() in ['M', 'MALE']:
                sex_context = (
                    "\n\nSEX-DIAGNOSIS RELEVANCE EVALUATION:\n"
                    "- Patient is MALE.\n"
                    "- Your task: Evaluate how RELEVANT and CONSISTENT the diagnosis is with MALE sex.\n"
                    "- SCORING PRINCIPLE: Higher score = MORE RELEVANT/CONSISTENT with this sex.\n\n"
                    "SCORING GUIDE (based on relevance/consistency):\n"
                    "- Score 8-10: Diagnosis is HIGHLY RELEVANT to males OR sex-neutral (most common)\n"
                    "  * Examples: Hemophilia A (X-linked, more common in males) → Score 9-10\n"
                    "  * Examples: Muscular injuries, hematomas, fractures → Score 8-9 (sex-neutral, very common)\n"
                    "  * Examples: Infections, pneumonia, common diseases → Score 8-9 (sex-neutral)\n"
                    "  * Examples: Most chronic conditions (diabetes, hypertension) → Score 8-9 (sex-neutral)\n"
                    "- Score 6-7: Diagnosis is MODERATELY relevant or somewhat more common in opposite sex\n"
                    "- Score 4-5: Diagnosis is LESS common in males but still possible\n"
                    "- Score 1-3: Diagnosis is FEMALE-SPECIFIC or VERY RARE in males\n"
                    "  * Examples: Ovarian cancer, cervical cancer → Score 1-2\n"
                    "  * Examples: Pregnancy-related conditions → Score 1-2\n"
                    "  * Examples: Female reproductive system diseases → Score 1-2\n\n"
                    "IMPORTANT:\n"
                    "- DEFAULT to 8-9 for sex-neutral conditions (this is MOST medical conditions)\n"
                    "- Only give low scores (1-3) if diagnosis is CLEARLY female-specific\n"
                    "- When uncertain → Default to 8 (most conditions are sex-neutral and relevant to both sexes)\n"
                )
            elif value.upper() in ['F', 'FEMALE']:
                sex_context = (
                    "\n\nSEX-DIAGNOSIS RELEVANCE EVALUATION:\n"
                    "- Patient is FEMALE.\n"
                    "- Your task: Evaluate how RELEVANT and CONSISTENT the diagnosis is with FEMALE sex.\n"
                    "- SCORING PRINCIPLE: Higher score = MORE RELEVANT/CONSISTENT with this sex.\n\n"
                    "SCORING GUIDE (based on relevance/consistency):\n"
                    "- Score 8-10: Diagnosis is HIGHLY RELEVANT to females OR sex-neutral (most common)\n"
                    "  * Examples: Autoimmune diseases (more common in females) → Score 9-10\n"
                    "  * Examples: Muscular injuries, hematomas, fractures → Score 8-9 (sex-neutral, very common)\n"
                    "  * Examples: Infections, pneumonia, common diseases → Score 8-9 (sex-neutral)\n"
                    "  * Examples: Most chronic conditions (diabetes, hypertension) → Score 8-9 (sex-neutral)\n"
                    "- Score 6-7: Diagnosis is MODERATELY relevant or somewhat more common in opposite sex\n"
                    "- Score 4-5: Diagnosis is LESS common in females but still possible\n"
                    "- Score 1-3: Diagnosis is MALE-SPECIFIC or VERY RARE in females\n"
                    "  * Examples: Prostate cancer → Score 1-2\n"
                    "  * Examples: Male reproductive system diseases → Score 1-2\n\n"
                    "IMPORTANT:\n"
                    "- DEFAULT to 8-9 for sex-neutral conditions (this is MOST medical conditions)\n"
                    "- Only give low scores (1-3) if diagnosis is CLEARLY male-specific\n"
                    "- When uncertain → Default to 8 (most conditions are sex-neutral and relevant to both sexes)\n"
                )
        else:
            raise ValueError("Invalid ConsistencyCheck type")
        
        text_clean = " ".join(clinical_text.split())
        truncated_text = text_clean[:2000] + "..." if len(text_clean) > 2000 else text_clean
        if not USE_FULL_TEXT:
            truncated_text = ""
        
        # 根据年龄判断是否需要特别严格
        age_context = ""
        extreme_age_warning = ""
        normal_age_guidance = ""
        if check_type == ConsistencyCheck.AGE:
            try:
                age_int = int(value)
                if age_int < 2:
                    # 极端年龄：0-1岁（婴儿）
                    age_context = "\n\n🚨 EXTREMELY CRITICAL: Patient is an INFANT (age 0-1 years)."
                    extreme_age_warning = (
                        "\n\n🚨 EXTREME INFANT AGE WARNING (MANDATORY LOW SCORE):\n"
                        "- Infants (0-1 years) CANNOT participate in ANY high-risk activities.\n"
                        "- Infants CANNOT: snowboard, ski, skateboard, drive, work, or engage in adult activities.\n"
                        "- If the text mentions ANY high-risk activity (snowboarding, skiing, etc.) with an infant, "
                        "this is PHYSICALLY IMPOSSIBLE → Score 1 (ONLY).\n"
                        "- Infants have COMPLETELY DIFFERENT disease patterns:\n"
                        "  * They cannot have activity-related injuries from sports\n"
                        "  * They cannot have occupation-related conditions\n"
                        "  * Most adult-onset conditions are IMPOSSIBLE in infants\n"
                        "- Examples of IMPOSSIBLE combinations (MUST score 1):\n"
                        "  * Age 0-1 + snowboarding/skiing injury → Score 1 (IMPOSSIBLE)\n"
                        "  * Age 0-1 + high-risk sport + any diagnosis → Score 1 (IMPOSSIBLE)\n"
                        "  * Age 0-1 + adult activity mention → Score 1 (IMPOSSIBLE)\n"
                        "- Even if diagnosis is medically possible (e.g., hemophilia), if combined with "
                        "high-risk activity mention, MUST score 1 (the activity is impossible for this age).\n"
                        "- DEFAULT for infants with activity mentions: Score 1 (ONLY).\n"
                    )
                elif age_int < 10:
                    age_context = "\n\n⚠️ CRITICAL: Patient is a YOUNG CHILD (age 2-9 years)."
                    extreme_age_warning = (
                        "\n\nEXTREME AGE WARNING:\n"
                        "- Children have COMPLETELY DIFFERENT disease patterns than adults.\n"
                        "- Many adult conditions are RARE or IMPOSSIBLE in children.\n"
                        "- High-risk activities (snowboarding, extreme sports) are UNUSUAL for children <10.\n"
                        "- Chronic conditions requiring years to develop are UNLIKELY in very young children.\n"
                        "- If the diagnosis mentions activities, exposures, or conditions typical of adults, "
                        "this is HIGHLY SUSPICIOUS and should score 1-2.\n"
                        "- Examples of IMPLAUSIBLE combinations:\n"
                        "  * Young child + high-risk sport injury + chronic condition = Score 1-2\n"
                        "  * Young child + adult-onset disease = Score 1-3\n"
                        "  * Young child + occupation-related condition = Score 1-2\n"
                    )
                elif age_int > 85:
                    age_context = "\n\n⚠️ CRITICAL: Patient is VERY OLD (age > 85)."
                    extreme_age_warning = (
                        "\n\nEXTREME AGE WARNING:\n"
                        "- Very elderly patients have different disease patterns.\n"
                        "- Some conditions are less common or present differently in advanced age.\n"
                        "- Be STRICT when evaluating consistency.\n"
                    )
                else:
                    # 正常年龄范围（10-85岁）
                    normal_age_guidance = (
                        "\n\nNORMAL AGE RANGE (10-85 years):\n"
                        "- Patient is in a NORMAL age range for adult medical conditions.\n"
                        "- Most common adult conditions are HIGHLY RELEVANT for this age group.\n"
                        "- Examples of HIGHLY RELEVANT conditions (should score 8-10):\n"
                        "  * Chronic conditions (diabetes, hypertension, heart disease) → Score 9-10\n"
                        "  * Age-related conditions (portal hypertension, ascites, liver disease) → Score 9-10\n"
                        "  * Common adult diseases → Score 8-9\n"
                        "  * Most medical conditions seen in adults → Score 8-9\n"
                        "- Only give LOW scores (1-3) if:\n"
                        "  * Diagnosis is clearly pediatric-specific (e.g., childhood cancers, developmental disorders)\n"
                        "  * Diagnosis requires very specific age ranges outside this patient's age\n"
                        "- DEFAULT: For normal age + common adult diagnosis → Score 8-9\n"
                    )
            except:
                pass
        
        # 检查文本中是否有高风险活动或成人活动
        activity_warning = ""
        if check_type == ConsistencyCheck.AGE:
            high_risk_activities = ['snowboarding', 'skiing', 'skateboarding', 'motorcycle', 
                                   'driving', 'work', 'occupation', 'job', 'employment']
            if any(activity in truncated_text.lower() for activity in high_risk_activities):
                try:
                    age_int = int(value)
                    if age_int < 2:
                        # 0-1岁婴儿 + 高风险活动 = 完全不可能
                        activity_warning = (
                            "\n\n🚨 IMPOSSIBLE ACTIVITY DETECTED:\n"
                            "- The text mentions high-risk activities (snowboarding, skiing, etc.).\n"
                            "- Patient is an INFANT (age 0-1 years).\n"
                            "- Infants CANNOT participate in these activities - this is PHYSICALLY IMPOSSIBLE.\n"
                            "- MANDATORY SCORE: 1 (ONLY). Do NOT give any other score.\n"
                            "- This combination is IMPOSSIBLE regardless of diagnosis.\n"
                        )
                    elif age_int < 10:
                        # 2-9岁儿童 + 高风险活动 = 高度不合理
                        activity_warning = (
                            "\n\n⚠️ ACTIVITY WARNING:\n"
                            "- The text mentions high-risk activities or adult activities.\n"
                            "- Patient is a young child (age 2-9 years).\n"
                            "- This is HIGHLY IMPLAUSIBLE for children this age.\n"
                            "- Score 1-2 for young children with high-risk activity mentions.\n"
                        )
                except:
                    activity_warning = (
                        "\n\n⚠️ ACTIVITY WARNING:\n"
                        "- The text mentions high-risk activities or adult activities.\n"
                        "- If patient is a young child (<10), this is HIGHLY IMPLAUSIBLE.\n"
                        "- Score 1-2 for young children with high-risk activity mentions.\n"
                    )
        
        # 构建prompt
        if check_type == ConsistencyCheck.SEX:
            prompt = (
                f"You are an EXTREMELY STRICT medical auditor. Your job is to identify IMPLAUSIBLE clinical combinations.\n\n"
                "TASK:\n"
                f"Patient {check_type.value.title()}: {value}\n"
                f"Primary Diagnosis: {diagnosis}\n"
                f"Discharge Summary Snippet: {truncated_text}\n\n"
                f"{sex_context}\n"
            )
        else:
            prompt = (
                f"You are a medical auditor. Your job is to evaluate age-diagnosis RELEVANCE.\n\n"
                "TASK:\n"
                f"Patient {check_type.value.title()}: {value}\n"
                f"Primary Diagnosis: {diagnosis}\n"
                f"Discharge Summary Snippet: {truncated_text}\n\n"
                f"{age_context}"
                f"{extreme_age_warning}"
                f"{normal_age_guidance}"
                f"{activity_warning}\n"
            )
        
        # 添加通用指令
        if check_type == ConsistencyCheck.AGE:
            prompt += (
                "INSTRUCTIONS:\n"
                f"- Evaluate based on {criteria}.\n"
                "- SCORING PRINCIPLE: Higher score = MORE RELEVANT/CONSISTENT with this age.\n"
                "- Score 9-10: Diagnosis is HIGHLY RELEVANT and VERY COMMON for this age group\n"
                "  * Typical age-appropriate conditions (most adult conditions) → Score 9-10\n"
                "  * Common chronic conditions (diabetes, hypertension, liver disease, etc.) → Score 9-10\n"
                "  * Age-related conditions (portal hypertension, ascites, etc.) → Score 9-10\n"
                "- Score 8: Diagnosis is RELEVANT and COMMON for this age group\n"
                "  * Most adult medical conditions → Score 8\n"
                "- Score 6-7: Diagnosis is MODERATELY relevant or somewhat less common for this age\n"
                "- Score 4-5: Diagnosis is LESS common for this age but still possible\n"
                "- Score 1-3: Diagnosis is HIGHLY IMPLAUSIBLE for this age\n"
                "  * Examples: Young child (<10) + high-risk sport injury + chronic condition → Score 1-2\n"
                "  * Examples: Young child + adult-onset disease → Score 1-3\n"
                "  * Examples: Very old (>85) + pediatric condition → Score 1-3\n"
                "- IMPORTANT: For patients aged 18-85:\n"
                "  * Most adult conditions are HIGHLY RELEVANT → Score 8-10\n"
                "  * Chronic diseases, age-related conditions → Score 9-10\n"
                "  * Only give low scores if diagnosis is clearly age-inappropriate\n"
                "- When uncertain → Default to 8-9 for normal age ranges with common adult conditions\n"
            )
        else:
            prompt += (
                "INSTRUCTIONS:\n"
                f"- Evaluate based on {criteria}.\n"
                "- SCORING PRINCIPLE: Higher score = MORE RELEVANT/CONSISTENT with this sex.\n"
            )
        
        # 添加通用结尾
        prompt += (
            "- IMPORTANT: For sex checks - RELEVANCE SCORING (READ CAREFULLY):\n"
            "  * Score based on RELEVANCE: How relevant/consistent is this diagnosis with this sex?\n"
            "  * MOST medical conditions are SEX-NEUTRAL → Score 8-9 (this is the DEFAULT)\n"
            "  * Higher score = More relevant/consistent with this sex\n"
            "  * Only give LOW scores (1-3) if diagnosis is CLEARLY sex-specific to the OPPOSITE sex\n"
            "  * Examples of sex-neutral (score 8-9): hemophilia, injuries, infections, most chronic diseases\n"
            "  * Examples requiring low score (1-2): prostate cancer in female, ovarian cancer in male\n"
            "  * When uncertain → Default to 8 (most conditions are sex-neutral and highly relevant to both sexes)\n"
            "- When in doubt about RELEVANCE → Consider if the diagnosis is typical and relevant for this patient's characteristics.\n"
            "- Do not provide explanations, notes, or labels.\n"
            "- Return ONLY the integer.\n\n"
            "SCORE:"
        )
        
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(retries + 1):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    options={"temperature": 0.2, "num_predict": 10}
                )
                score_str = response["message"]["content"].strip()
                match = re.search(r"\b([1-9]|10)\b", score_str)
                if match:
                    return int(match.group(1))
                
                messages.append({"role": "assistant", "content": score_str})
                messages.append({"role": "user", "content": "You did not output an integer. Please output ONLY the integer score (1-10)."})
            except Exception as e:
                print(f"API Error on {check_type.value} check attempt {attempt+1}: {e}")
                time.sleep(2)
        return -1


def calculate_age_at_discharge(text: str) -> int:
    """从文本中计算年龄"""
    discharge_pattern = r"Discharge Date:\s*([0-9\-\/]+)"
    dob_pattern = r"(?:Date of Birth|DOB):\s*([0-9\-\/]+)"
    try:
        discharge_match = re.search(discharge_pattern, text, re.IGNORECASE)
        dob_match = re.search(dob_pattern, text, re.IGNORECASE)
        if not discharge_match or not dob_match:
            return -1
        
        discharge_str = discharge_match.group(1).strip()
        dob_str = dob_match.group(1).strip()
        discharge_date = parser.parse(discharge_str)
        dob_date = parser.parse(dob_str)
        
        age = discharge_date.year - dob_date.year - (
            (discharge_date.month, discharge_date.day) < (dob_date.month, dob_date.day)
        )
        return age
    except (ValueError, OverflowError):
        return -1


def extract_sex(text: str) -> Optional[str]:
    """从文本中提取性别"""
    sex_pattern = r"(?:Sex|Gender):\s*([A-Za-z]+)"
    match = re.search(sex_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_diagnosis(text: str) -> Optional[str]:
    """从文本中提取诊断"""
    pattern = r"Discharge Diagnosis:\s*([\s\S]+?)(?=\n\s*\n|Discharge Condition:|History|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace('\n', ' ')
    return "Unknown Diagnosis"


# ============================================================
# 主处理函数
# ============================================================

def process_single_row(row_data: Tuple[int, dict]) -> dict:
    """
    Process single row of data, integrate logic from all three modules
    """
    index, row = row_data
    
    # 初始化结果
    result = {
        "index": index,
        "extracted_text": row.get("extracted_text", ""),
        "filled_text": row.get("filled_text", ""),
        "phi_annotations": row.get("phi_annotations", "[]"),
        "score_age": 0.0,
        "score_sex": 0.0,
        "clinical_consistency_realism": 0.0,
        "document_structure_realism": 0.0,
        "document_structure_score_reason": "",
        "linguistic_realism": 0.0,
        "phi_amount_score": 0.0,
        "ambiguity_score": 0.0,
        "DIFFICULTY": 0.0,
        "realism": 0.0,
        "overall": "",
        "error": None
    }
    
    # Save bad_phi_details (if input has it)
    if "bad_phi_details" in row:
        result["bad_phi_details"] = row.get("bad_phi_details", "[]")
    
    filled_text = row.get("filled_text", "")
    extracted_text = row.get("extracted_text", "")
    phi_annotations_str = row.get("phi_annotations", "[]")
    
    # Parse phi_annotations
    try:
        phi_annotations = json.loads(phi_annotations_str)
        if not isinstance(phi_annotations, list):
            phi_annotations = []
    except:
        phi_annotations = []
    
    # MODULE 1: Document Structure
    if extracted_text or filled_text:
        text_for_structure = filled_text if filled_text else extracted_text
        doc_score, doc_reason = calculate_document_structure_score(text_for_structure)
        result["document_structure_realism"] = doc_score
        result["document_structure_score_reason"] = doc_reason
    
    # MODULE 2: Linguistic & Difficulty
    linguistic_result = compute_linguistic_for_row(phi_annotations)
    if isinstance(linguistic_result, tuple):
        linguistic, linguistic_details = linguistic_result
    else:
        # 兼容旧版本（如果返回单个值）
        linguistic = linguistic_result
        linguistic_details = []
    phi_amount, ambiguity, difficulty = compute_difficulty_for_row(phi_annotations)
    result["linguistic_realism"] = linguistic
    result["linguistic_details"] = linguistic_details  # 保存详细信息
    result["phi_amount_score"] = phi_amount
    result["ambiguity_score"] = ambiguity
    result["DIFFICULTY"] = difficulty
    
    # MODULE 3: Clinical Consistency (check age and sex)
    if filled_text:
        age = calculate_age_at_discharge(filled_text)
        sex = extract_sex(filled_text)
        diagnosis = extract_diagnosis(filled_text)
        
        if age != -1 and sex and diagnosis:
            try:
                age_score = ClinicalConsistencyEvaluator.get_relevance(
                    ConsistencyCheck.AGE,
                    value=str(age),
                    diagnosis=diagnosis,
                    clinical_text=filled_text
                )
                sex_score = ClinicalConsistencyEvaluator.get_relevance(
                    ConsistencyCheck.SEX,
                    value=sex,
                    diagnosis=diagnosis,
                    clinical_text=filled_text
                )
                
                if age_score != -1 and sex_score != -1:
                    # Divide by 10 (convert from 1-10 score to 0-1)
                    age_score_norm = age_score / 10.0
                    sex_score_norm = sex_score / 10.0
                    result["score_age"] = age_score_norm
                    result["score_sex"] = sex_score_norm
                    
                    # Improved clinical_consistency_realism calculation
                    # If age_score is very low, directly heavily penalize (because there are only two factors)
                    import math
                    
                    # If age_score is very low, directly significantly reduce clinical_consistency
                    if age_score_norm < 0.3:
                        # Very severe: age score extremely low (< 0.3), directly significantly reduce
                        # Example: age=0.2, sex=0.8 → clinical directly reduced to 0.2-0.3
                        result["clinical_consistency_realism"] = age_score_norm * 1.2  # Slightly higher than age, but very low
                    elif age_score_norm < 0.5:
                        # Severe: age score low (< 0.5), significantly reduce
                        # Example: age=0.4, sex=0.8 → clinical reduced to 0.3-0.4
                        # Use geometric mean to make low scores have greater impact
                        age_for_calc = max(0.01, age_score_norm)
                        sex_for_calc = max(0.01, sex_score_norm)
                        geometric_mean = math.sqrt(age_for_calc * sex_for_calc)
                        result["clinical_consistency_realism"] = geometric_mean * 0.6  # Significantly reduce
                    else:
                        # age_score normal (>= 0.5), use weighted average (age weight slightly higher)
                        result["clinical_consistency_realism"] = (age_score_norm * 0.6 + sex_score_norm * 0.4)
                else:
                    result["error"] = "LLM returned -1"
            except Exception as e:
                result["error"] = f"Clinical consistency error: {str(e)}"
        else:
            result["error"] = "Failed to extract age/sex/diagnosis"
    
    # Calculate final realism - use improved formula
    # Option 1: Weighted geometric mean (stricter, any dimension being low will significantly reduce total score)
    # Option 2: Weighted average + penalty term (additional penalty for low scores)
    
    # Use weighted geometric mean to make low scores have greater impact
    ling = max(0.01, result["linguistic_realism"])  # Avoid 0 causing geometric mean to be 0
    clin = max(0.01, result["clinical_consistency_realism"])
    struct = max(0.01, result["document_structure_realism"])
    
    # Weighted geometric mean: any item being low will significantly reduce total score
    # Formula: (ling^w1 * clin^w2 * struct^w3) ^ (1/(w1+w2+w3))
    # But to maintain weight meaning, we use weighted average in log space
    import math
    
    # Use weighted geometric mean
    log_realism = (
        LINGUISTIC_WEIGHT * math.log(ling) +
        CLINICAL_WEIGHT * math.log(clin) +
        STRUCTURE_WEIGHT * math.log(struct)
    )
    geometric_mean = math.exp(log_realism)
    
    # Mixed approach: 70% geometric mean + 30% arithmetic mean
    # This allows low scores to have greater impact while not being too extreme
    arithmetic_mean = (
        LINGUISTIC_WEIGHT * result["linguistic_realism"] +
        CLINICAL_WEIGHT * result["clinical_consistency_realism"] +
        STRUCTURE_WEIGHT * result["document_structure_realism"]
    )
    
    result["realism"] = 0.7 * geometric_mean + 0.3 * arithmetic_mean
    
    # Analyze deduction reasons, generate overall description
    overall_issues = []
    detailed_info = []
    
    # 1. Linguistic analysis
    ling_score = result["linguistic_realism"]
    linguistic_details = result.get("linguistic_details", [])
    
    if ling_score < 0.3:
        overall_issues.append("LINGUISTIC严重问题")
        if linguistic_details and linguistic_details != ["无违反"]:
            detailed_info.append(f"LING={ling_score:.2f}(严重:{';'.join(linguistic_details)})")
        else:
            detailed_info.append(f"LING={ling_score:.2f}(严重)")
    elif ling_score < 0.5:
        overall_issues.append("LINGUISTIC有问题")
        if linguistic_details and linguistic_details != ["无违反"]:
            detailed_info.append(f"LING={ling_score:.2f}(有问题:{';'.join(linguistic_details)})")
        else:
            detailed_info.append(f"LING={ling_score:.2f}(有问题)")
    elif ling_score < 0.7:
        overall_issues.append("LINGUISTIC轻微问题")
        if linguistic_details and linguistic_details != ["无违反"]:
            detailed_info.append(f"LING={ling_score:.2f}(轻微:{';'.join(linguistic_details)})")
        else:
            detailed_info.append(f"LING={ling_score:.2f}(轻微)")
    else:
        if linguistic_details and linguistic_details != ["无违反"]:
            detailed_info.append(f"LING={ling_score:.2f}(正常:{';'.join(linguistic_details)})")
        else:
            detailed_info.append(f"LING={ling_score:.2f}(正常)")
    
    # 2. Clinical Consistency analysis
    clin_score = result["clinical_consistency_realism"]
    age_score = result.get("score_age", 0.0)
    sex_score = result.get("score_sex", 0.0)
    
    if clin_score < 0.3:
        if age_score < 0.3:
            overall_issues.append("CLINICAL严重问题:年龄-诊断不匹配")
            detailed_info.append(f"CLIN={clin_score:.2f}(严重,age={age_score:.2f})")
        elif age_score < 0.5:
            overall_issues.append("CLINICAL严重问题:年龄-诊断不合理")
            detailed_info.append(f"CLIN={clin_score:.2f}(严重,age={age_score:.2f})")
        else:
            overall_issues.append("CLINICAL严重问题")
            detailed_info.append(f"CLIN={clin_score:.2f}(严重)")
    elif clin_score < 0.5:
        if age_score < 0.5:
            overall_issues.append("CLINICAL有问题:年龄-诊断不太合理")
            detailed_info.append(f"CLIN={clin_score:.2f}(有问题,age={age_score:.2f})")
        else:
            overall_issues.append("CLINICAL有问题")
            detailed_info.append(f"CLIN={clin_score:.2f}(有问题)")
    elif clin_score < 0.7:
        overall_issues.append("CLINICAL轻微问题")
        detailed_info.append(f"CLIN={clin_score:.2f}(轻微)")
    else:
        detailed_info.append(f"CLIN={clin_score:.2f}(正常)")
    
    # 3. Document Structure analysis
    struct_score = result["document_structure_realism"]
    if struct_score < 0.3:
        overall_issues.append("STRUCTURE严重问题")
        detailed_info.append(f"STRUCT={struct_score:.2f}(严重)")
    elif struct_score < 0.5:
        overall_issues.append("STRUCTURE有问题")
        detailed_info.append(f"STRUCT={struct_score:.2f}(有问题)")
    elif struct_score < 0.7:
        overall_issues.append("STRUCTURE轻微问题")
        detailed_info.append(f"STRUCT={struct_score:.2f}(轻微)")
    else:
        detailed_info.append(f"STRUCT={struct_score:.2f}(正常)")
    
    # 4. Find lowest dimension (main deduction item)
    scores = {
        "LINGUISTIC": ling_score,
        "CLINICAL": clin_score,
        "STRUCTURE": struct_score
    }
    min_dimension = min(scores.items(), key=lambda x: x[1])
    
    # 5. Generate overall description
    if len(overall_issues) == 0:
        result["overall"] = "No major issues | " + " | ".join(detailed_info)
    else:
        # Main deduction item
        main_issue = f"Main deduction:{min_dimension[0]}({min_dimension[1]:.2f})"
        # Other issues (excluding main deduction item)
        other_issues = [issue for issue in overall_issues if min_dimension[0] not in issue]
        if other_issues:
            result["overall"] = f"{main_issue} | Others:{';'.join(other_issues)} | " + " | ".join(detailed_info)
        else:
            result["overall"] = f"{main_issue} | " + " | ".join(detailed_info)
    
    return result


def process_csv(input_path: str, output_path: str, max_rows: Optional[int] = None):
    """
    Main processing function: read CSV, process each row, output results
    """
    print(f"Reading {input_path}...")
    
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if max_rows is not None:
        print(f"Limiting to first {max_rows} rows")
        rows = rows[:max_rows]
    
    print(f"Total {len(rows)} rows, using {MAX_WORKERS} threads to process...")
    
    # Prepare data
    rows_to_process = list(enumerate(rows))
    results_list = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(process_single_row, row): row for row in rows_to_process}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_row)):
            result = future.result()
            results_list.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(rows)} rows...")
    
    print(f"Processing complete, took {time.time() - start_time:.2f} seconds")
    
    # Sort by index
    results_list.sort(key=lambda x: x["index"])
    
    # Output columns
    output_columns = [
        'extracted_text',
        'filled_text',
        'phi_annotations',
        'score_age',
        'score_sex',
        'clinical_consistency_realism',
        'document_structure_realism',
        'document_structure_score_reason',
        'linguistic_realism',
        'phi_amount_score',
        'ambiguity_score',
        'DIFFICULTY',
        'realism',
        'overall'
    ]
    
    # If input has bad_phi_details, also add to output
    if rows and "bad_phi_details" in rows[0]:
        output_columns.append('bad_phi_details')
    
    # Write to output file (filter out invalid rows)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        
        valid_count = 0
        skipped_count = 0
        
        for result in results_list:
            # Check if invalid row (all scores are 0 or close to 0)
            try:
                linguistic = float(result.get("linguistic_realism", 0))
                clinical = float(result.get("clinical_consistency_realism", 0))
                structure = float(result.get("document_structure_realism", 0))
                realism = float(result.get("realism", 0))
            except (ValueError, TypeError):
                # If cannot convert to number, skip
                skipped_count += 1
                continue
            
            # If all scores are close to 0 (less than 0.01), skip this row
            if (linguistic < 0.01 and clinical < 0.01 and structure < 0.01 and realism < 0.01):
                skipped_count += 1
                continue
            
            # Only write required columns
            output_row = {col: result.get(col, "") for col in output_columns}
            writer.writerow(output_row)
            valid_count += 1
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} rows of invalid data (all scores are 0 or close to 0)")
    
    print(f"\n✓ Successfully created {output_path}")
    print(f"Input rows: {len(results_list)}")
    print(f"Valid output rows: {valid_count}")
    
    # Show example
    if results_list:
        print("\nFirst row example:")
        first = results_list[0]
        print(f"  linguistic_realism: {first['linguistic_realism']:.4f}")
        print(f"  clinical_consistency_realism: {first['clinical_consistency_realism']:.4f}")
        print(f"  document_structure_realism: {first['document_structure_realism']:.4f}")
        print(f"  realism: {first['realism']:.4f}")
        if first.get('error'):
            print(f"  error: {first['error']}")


def main():
    """Main function"""
    print("=" * 60)
    print("Unified Preprocessing Script - Generate preprocess_data_label.csv")
    print("=" * 60)
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"LLM model: {MODEL_NAME}")
    print(f"Concurrent threads: {MAX_WORKERS}")
    print(f"Realism weights: Linguistic={LINGUISTIC_WEIGHT}, Clinical={CLINICAL_WEIGHT}, Structure={STRUCTURE_WEIGHT}")
    print("=" * 60)
    
    # Set max_rows=None to process all data, or set a number for testing
    process_csv(INPUT_CSV, OUTPUT_CSV, max_rows=None)


if __name__ == "__main__":
    main()

