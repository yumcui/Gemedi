"""
Enhanced PHI Data Filling Tool with LLM Support

This script improves upon fill_phi_data.py by using LLM to generate
more realistic and diverse PHI values (especially hospital names, facility names)
while maintaining the exact format preservation logic of the original script.
"""

import csv
import json
import re
from datetime import datetime, timedelta
from faker import Faker
import random
import ollama
from typing import Optional, Dict, List

# 初始化Faker生成器
fake = Faker()
Faker.seed(42)  # 设置种子确保可复现性
random.seed(42)

# LLM Configuration
MODEL_NAME = "llama3.1"  # Change this to your preferred model
USE_LLM = True  # Set to False to fall back to Faker only

# Probability settings for LLM vs Faker
LLM_PROBABILITY_PATIENT_NAME = 0.5  # 50% chance to use LLM for patient names
LLM_PROBABILITY_DOCTOR_NAME = 0.5   # 50% chance to use LLM for doctor names
LLM_PROBABILITY_HOSPITAL = 0.5       # 50% chance to use LLM for hospital names
LLM_PROBABILITY_FACILITY = 0.5      # 50% chance to use LLM for facility names
LLM_PROBABILITY_FOLLOWUP = 1.0      # 100% use LLM for follow-up instructions (must use attending doctor)


class LLMPHIGenerator:
    """使用LLM生成更真实的PHI值"""
    
    @staticmethod
    def generate_hospital_name(context: str = "") -> str:
        """使用LLM生成医院名称"""
        if not USE_LLM:
            # Fallback to Faker
            return f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital', 'Community Hospital'])}"
        
        prompt = f"""Generate a realistic synthetic hospital name. It should sound like a real hospital but be completely fictional.

Examples of good hospital names:
- "Riverside Medical Center"
- "Memorial General Hospital" 
- "University of [City] Medical Center"
- "St. [Name] Hospital"
- "[City] Regional Health Center"

Generate ONE hospital name only (no quotes, no explanation, just the name):"""
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8, "num_predict": 50}
            )
            hospital_name = response["message"]["content"].strip()
            # Clean up any quotes or extra text
            hospital_name = re.sub(r'^["\']|["\']$', '', hospital_name)
            hospital_name = hospital_name.split('\n')[0].strip()
            if not hospital_name:
                raise ValueError("Empty response")
            return hospital_name
        except Exception as e:
            print(f"Warning: LLM failed for hospital name: {e}, using Faker fallback")
            return f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital'])}"
    
    @staticmethod
    def generate_facility_name(context: str = "") -> str:
        """使用LLM生成医疗设施名称"""
        if not USE_LLM:
            # Fallback to Faker
            facilities = [
                "Home Health Services", "Visiting Nurse Association",
                "Community Health Services", "Home Care Partners"
            ]
            return random.choice(facilities)
        
        prompt = f"""Generate a realistic synthetic home health care facility name. It should sound like a real facility but be completely fictional.

Examples of good facility names:
- "Riverside Home Health Services"
- "Community Care Partners"
- "Premier Home Health Care"
- "Compassionate Care Services"

Generate ONE facility name only (no quotes, no explanation, just the name):"""
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8, "num_predict": 50}
            )
            facility_name = response["message"]["content"].strip()
            facility_name = re.sub(r'^["\']|["\']$', '', facility_name)
            facility_name = facility_name.split('\n')[0].strip()
            if not facility_name:
                raise ValueError("Empty response")
            return facility_name
        except Exception as e:
            print(f"Warning: LLM failed for facility name: {e}, using Faker fallback")
            facilities = ["Home Health Services", "Visiting Nurse Association", "Community Health Services"]
            return random.choice(facilities)
    
    @staticmethod
    def generate_patient_name_llm(sex: Optional[str] = None) -> str:
        """使用LLM生成患者姓名"""
        sex_hint = ""
        if sex == 'F':
            sex_hint = " (female name)"
        elif sex == 'M':
            sex_hint = " (male name)"
        
        prompt = f"""Generate a realistic synthetic patient name. Format: "[First Name] [Last Name]"{sex_hint}

The name should sound realistic but be completely fictional.
Generate ONE patient name in the format "FirstName LastName" (no quotes, no explanation):"""
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 50}
            )
            name = response["message"]["content"].strip()
            name = re.sub(r'^["\']|["\']$', '', name)
            name = name.split('\n')[0].strip()
            if not name or len(name.split()) < 2:
                raise ValueError("Invalid format")
            return name
        except Exception as e:
            print(f"Warning: LLM failed for patient name: {e}, using Faker fallback")
            if sex == 'F':
                return f"{fake.first_name_female()} {fake.last_name()}"
            elif sex == 'M':
                return f"{fake.first_name_male()} {fake.last_name()}"
            else:
                return f"{fake.first_name()} {fake.last_name()}"
    
    @staticmethod
    def generate_doctor_name(sex: Optional[str] = None, context: str = "") -> str:
        """使用LLM生成医生姓名（更真实的组合）"""
        if not USE_LLM:
            # Fallback to Faker
            if sex == 'F':
                return f"Dr. {fake.first_name_female()} {fake.last_name()}"
            elif sex == 'M':
                return f"Dr. {fake.first_name_male()} {fake.last_name()}"
            else:
                return f"Dr. {fake.first_name()} {fake.last_name()}"
        
        prompt = f"""Generate a realistic synthetic doctor name. Format: "Dr. [First Name] [Last Name]"

The name should sound professional and realistic but be completely fictional.
Generate ONE doctor name in the exact format "Dr. FirstName LastName" (no quotes, no explanation):"""
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 50}
            )
            doctor_name = response["message"]["content"].strip()
            doctor_name = re.sub(r'^["\']|["\']$', '', doctor_name)
            doctor_name = doctor_name.split('\n')[0].strip()
            # Ensure it starts with "Dr. "
            if not doctor_name.startswith("Dr. "):
                doctor_name = "Dr. " + doctor_name
            if not doctor_name or len(doctor_name.split()) < 3:
                raise ValueError("Invalid format")
            return doctor_name
        except Exception as e:
            print(f"Warning: LLM failed for doctor name: {e}, using Faker fallback")
            if sex == 'F':
                return f"Dr. {fake.first_name_female()} {fake.last_name()}"
            elif sex == 'M':
                return f"Dr. {fake.first_name_male()} {fake.last_name()}"
            else:
                return f"Dr. {fake.first_name()} {fake.last_name()}"
    
    @staticmethod
    def generate_followup_instructions(attending_doctor: str, patient_context: str = "") -> str:
        """使用LLM生成更真实的随访说明"""
        if not USE_LLM:
            # Fallback to simple template
            return f"Follow up with {attending_doctor} in 2 weeks"
        
        # 提取医生名字（去掉 "Dr. " 前缀以便在 prompt 中更灵活）
        doctor_name = attending_doctor.replace("Dr. ", "").strip()
        
        prompt = f"""Generate realistic follow-up instructions for a patient. The instructions should be natural, professional, and specific.

IMPORTANT: You MUST use the exact attending physician name provided below. Do NOT generate a different doctor name.

Attending physician: {attending_doctor}
Patient context: {patient_context[:200] if patient_context else "General medical follow-up"}

Generate realistic follow-up instructions that MUST mention "{attending_doctor}" (use this exact name). 

Examples of good follow-up instructions:
- "Follow up with {attending_doctor} in 2 weeks for routine check-up"
- "Schedule a follow-up appointment with {attending_doctor} in 1-2 weeks to monitor progress"
- "Please follow up with {attending_doctor} in 2 weeks, or sooner if symptoms worsen"
- "Return to see {attending_doctor} in 2 weeks for re-evaluation"

CRITICAL: The follow-up instruction MUST include "{attending_doctor}" - do not use any other doctor name.

Generate ONE realistic follow-up instruction (1-2 sentences, no quotes, just the instruction text):"""
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8, "num_predict": 100}
            )
            followup = response["message"]["content"].strip()
            followup = re.sub(r'^["\']|["\']$', '', followup)
            followup = followup.split('\n')[0].strip()
            if not followup:
                raise ValueError("Empty response")
            return followup
        except Exception as e:
            print(f"Warning: LLM failed for follow-up instructions: {e}, using template fallback")
            return f"Follow up with {attending_doctor} in 2 weeks"


class PHIGenerator:
    """生成医疗PHI（受保护健康信息）数据"""

    def __init__(self):
        self.current_patient = {}
        self.phi_annotations = []
        self.llm_generator = LLMPHIGenerator()

    def reset_patient(self):
        """为新患者重置数据"""
        self.current_patient = {}
        self.phi_annotations = []

    def generate_patient_name(self, sex=None):
        """生成患者姓名 - 50%概率使用LLM，50%概率使用Faker"""
        # 决定是否使用LLM
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_PATIENT_NAME
        
        if use_llm:
            try:
                full_name = self.llm_generator.generate_patient_name_llm(sex)
                # 解析名字
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:])
                else:
                    # Fallback to Faker if parsing fails
                    if sex == 'F':
                        first_name = fake.first_name_female()
                        last_name = fake.last_name()
                    elif sex == 'M':
                        first_name = fake.first_name_male()
                        last_name = fake.last_name()
                    else:
                        first_name = fake.first_name()
                        last_name = fake.last_name()
                    full_name = f"{first_name} {last_name}"
            except Exception as e:
                # Fallback to Faker
                if sex == 'F':
                    first_name = fake.first_name_female()
                    last_name = fake.last_name()
                elif sex == 'M':
                    first_name = fake.first_name_male()
                    last_name = fake.last_name()
                else:
                    first_name = fake.first_name()
                    last_name = fake.last_name()
                full_name = f"{first_name} {last_name}"
        else:
            # 使用Faker
            if sex == 'F':
                first_name = fake.first_name_female()
                last_name = fake.last_name()
            elif sex == 'M':
                first_name = fake.first_name_male()
                last_name = fake.last_name()
            else:
                first_name = fake.first_name()
                last_name = fake.last_name()
            full_name = f"{first_name} {last_name}"

        self.current_patient['name'] = full_name
        self.current_patient['first_name'] = first_name
        self.current_patient['last_name'] = last_name
        return full_name

    def generate_unit_no(self):
        """生成病历号"""
        unit_no = str(random.randint(1000000, 9999999))
        self.current_patient['unit_no'] = unit_no
        return unit_no

    def generate_dates(self):
        """生成入院和出院日期"""
        # 生成过去2年内的日期
        admission_date = fake.date_between(start_date='-2y', end_date='-30d')
        # 出院日期在入院后1-30天
        stay_days = random.randint(1, 30)
        discharge_date = admission_date + timedelta(days=stay_days)

        # 格式化日期
        admission_str = admission_date.strftime('%Y-%m-%d')
        discharge_str = discharge_date.strftime('%Y-%m-%d')

        self.current_patient['admission_date'] = admission_str
        self.current_patient['discharge_date'] = discharge_str

        return admission_str, discharge_str

    def generate_dob(self, sex=None):
        """生成出生日期（18-90岁）"""
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        dob_str = dob.strftime('%Y-%m-%d')
        self.current_patient['dob'] = dob_str
        return dob_str

    def generate_attending_name(self):
        """生成主治医生姓名 - 50%概率使用LLM，50%概率使用Faker"""
        # 决定是否使用LLM
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_DOCTOR_NAME
        
        if use_llm:
            try:
                attending = self.llm_generator.generate_doctor_name()
            except Exception as e:
                # Fallback to Faker
                attending = f"Dr. {fake.first_name()} {fake.last_name()}"
        else:
            # 使用Faker
            attending = f"Dr. {fake.first_name()} {fake.last_name()}"
        
        self.current_patient['attending'] = attending
        return attending

    def generate_facility_name(self):
        """生成医疗机构名称 - 50%概率使用LLM，50%概率使用Faker"""
        # 决定是否使用LLM
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_FACILITY
        
        if use_llm:
            try:
                facility = self.llm_generator.generate_facility_name()
            except Exception as e:
                # Fallback to Faker
                facilities = [
                    "Home Health Services", "Visiting Nurse Association",
                    "Community Health Services", "Home Care Partners"
                ]
                facility = random.choice(facilities)
        else:
            # 使用Faker
            facilities = [
                "Home Health Services", "Visiting Nurse Association",
                "Community Health Services", "Home Care Partners"
            ]
            facility = random.choice(facilities)
        
        self.current_patient['facility'] = facility
        return facility

    def generate_hospital_name(self):
        """生成医院名称 - 50%概率使用LLM，50%概率使用Faker"""
        # 决定是否使用LLM
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_HOSPITAL
        
        if use_llm:
            try:
                hospital = self.llm_generator.generate_hospital_name()
            except Exception as e:
                # Fallback to Faker
                hospital = f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital'])}"
        else:
            # 使用Faker
            hospital = f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital'])}"
        
        self.current_patient['hospital'] = hospital
        return hospital

    def add_annotation(self, value, phi_type, reason):
        """添加PHI标注"""
        self.phi_annotations.append({
            "value": value,
            "type": phi_type,
            "reason": reason
        })

    def get_annotations_json(self):
        """返回JSON格式的标注"""
        return json.dumps(self.phi_annotations, ensure_ascii=False)


def fill_phi_in_text(text, original_col2):
    """填充文本中的所有PHI占位符 - 保持原有逻辑，使用LLM增强生成"""
    generator = PHIGenerator()

    # 首先提取性别信息（如果有）
    sex = None
    sex_match = re.search(r'Sex:\s*([MF])', text)
    if sex_match:
        sex = sex_match.group(1)

    # 按顺序处理PHI字段

    # 1. Name (患者姓名)
    name_pattern = r'Name:\s*___'
    if re.search(name_pattern, text):
        patient_name = generator.generate_patient_name(sex)
        text = re.sub(name_pattern, f'Name:  {patient_name}', text, count=1)
        generator.add_annotation(patient_name, "NAME", "Patient's name")

    # 2. Unit No (病历号)
    unit_pattern = r'Unit No:\s*___'
    if re.search(unit_pattern, text):
        unit_no = generator.generate_unit_no()
        text = re.sub(unit_pattern, f'Unit No:   {unit_no}', text, count=1)
        generator.add_annotation(unit_no, "UNIT_NO", "Patient's unit number")

    # 3. Admission Date and Discharge Date (入院和出院日期)
    admission_pattern = r'Admission Date:\s*___'
    discharge_pattern = r'Discharge Date:\s*___'
    if re.search(admission_pattern, text) or re.search(discharge_pattern, text):
        admission_date, discharge_date = generator.generate_dates()
        text = re.sub(admission_pattern, f'Admission Date:  {admission_date}', text)
        text = re.sub(discharge_pattern, f'Discharge Date:   {discharge_date}', text)

        if re.search(r'Admission Date:', text):
            generator.add_annotation(admission_date, "DATE_OF_ADMISSION", "Admission date")
        if re.search(r'Discharge Date:', text):
            generator.add_annotation(discharge_date, "DISCHARGE_DATE", "Discharge date")

    # 4. Date of Birth (出生日期)
    dob_pattern = r'Date of Birth:\s*___'
    if re.search(dob_pattern, text):
        dob = generator.generate_dob(sex)
        text = re.sub(dob_pattern, f'Date of Birth:  {dob}', text, count=1)
        generator.add_annotation(dob, "DATE_OF_BIRTH", "Patient's date of birth")

    # 5. Attending (主治医生) - 使用LLM增强
    attending_pattern = r'Attending:\s*___\.?'
    if re.search(attending_pattern, text):
        attending = generator.generate_attending_name()
        text = re.sub(attending_pattern, f'Attending: {attending}', text)
        generator.add_annotation(attending, "ATTENDING_PROVIDER", "Attending physician's name")

    # 6. Dear Mr./Ms. ___ (称呼中的姓名)
    dear_pattern = r'Dear (Mr\.|Ms\.)\s*___,'
    dear_match = re.search(dear_pattern, text)
    if dear_match:
        title = dear_match.group(1)
        if generator.current_patient.get('last_name'):
            last_name = generator.current_patient['last_name']
        else:
            # 如果还没有生成姓名，现在生成
            if title == 'Ms.':
                full_name = generator.generate_patient_name('F')
            else:
                full_name = generator.generate_patient_name('M')
            last_name = generator.current_patient['last_name']

        text = re.sub(dear_pattern, f'Dear {title} {last_name},', text)
        generator.add_annotation(title, "NAME_TITLE", "Name title")
        generator.add_annotation(last_name, "DISCHARGE_INSTRUCTIONS_NAME", "Name in discharge instructions")

    # 7. Dr. ___ (其他医生引用) - 使用LLM增强
    dr_pattern = r'Dr\.\s*___'
    dr_matches = list(re.finditer(dr_pattern, text))
    for match in dr_matches:
        if generator.current_patient.get('attending'):
            doctor_name = generator.current_patient['attending']
        else:
            doctor_name = generator.generate_attending_name()
        text = text[:match.start()] + doctor_name + text[match.end():]
        generator.add_annotation(doctor_name, "PROVIDER_NAME", "Provider name in text")

    # 8. Your ___ Team (医院/团队名称) - 使用LLM增强
    team_pattern = r'Your\s+___\s+Team'
    if re.search(team_pattern, text):
        hospital = generator.generate_hospital_name()
        text = re.sub(team_pattern, f'Your {hospital} Team', text)
        generator.add_annotation(hospital, "HOSPITAL_NAME", "Hospital name")

    # 9. Facility: ___ (医疗设施) - 使用LLM增强
    facility_pattern = r'Facility:\s*___'
    if re.search(facility_pattern, text):
        facility = generator.generate_facility_name()
        text = re.sub(facility_pattern, f'Facility:\n{facility}', text)
        generator.add_annotation(facility, "FACILITY_NAME", "Healthcare facility name")

    # 10. ___ at ___ (医院名称引用) - 使用LLM增强
    at_hospital_pattern = r'at\s+___[\.\s]'
    if re.search(at_hospital_pattern, text):
        hospital = generator.generate_hospital_name()
        text = re.sub(at_hospital_pattern, f'at {hospital}. ', text)
        generator.add_annotation(hospital, "HOSPITAL_NAME", "Hospital name reference")

    # 11. ___ Dementia or other diagnosis starting with ___
    diagnosis_start_pattern = r'^___\s+([A-Z][a-z]+)'
    diag_match = re.search(diagnosis_start_pattern, text, re.MULTILINE)
    if diag_match:
        # 常见诊断前缀
        diagnosis_prefixes = ["Lewy Body", "Vascular", "Alzheimer's", "Mixed", "Frontotemporal"]
        prefix = random.choice(diagnosis_prefixes)
        text = re.sub(diagnosis_start_pattern, f'{prefix} \\1', text, count=1)
        generator.add_annotation(prefix, "DIAGNOSIS_PREFIX", "Diagnosis type prefix")

    # 12. Followup Instructions: ___ (随访说明) - 100%使用LLM，必须使用已生成的attending医生
    followup_pattern = r'Followup Instructions:\s*___'
    if re.search(followup_pattern, text):
        # 确保attending医生已经生成（如果没有，现在生成）
        attending = generator.current_patient.get('attending')
        if not attending:
            attending = generator.generate_attending_name()
        
        # 提取一些上下文信息用于生成更真实的随访说明
        context_parts = []
        if 'Service:' in text:
            service_match = re.search(r'Service:\s*([^\n]+)', text)
            if service_match:
                context_parts.append(f"Service: {service_match.group(1).strip()}")
        if 'Discharge Diagnosis:' in text:
            diag_match = re.search(r'Discharge Diagnosis:\s*([^\n]+)', text)
            if diag_match:
                context_parts.append(f"Diagnosis: {diag_match.group(1).strip()}")
        patient_context = " | ".join(context_parts)
        
        # 100%使用LLM生成更真实的随访说明，必须使用attending医生
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_FOLLOWUP
        if use_llm:
            try:
                followup_text = generator.llm_generator.generate_followup_instructions(attending, patient_context)
            except Exception as e:
                # Fallback to template
                followup_text = f"Follow up with {attending} in 2 weeks"
        else:
            # Fallback to template
            followup_text = f"Follow up with {attending} in 2 weeks"
        
        text = re.sub(followup_pattern, f'Followup Instructions:\n{followup_text}', text)
        generator.add_annotation(followup_text, "FOLLOWUP_INSTRUCTIONS", "Follow-up instructions")

    # 13. 处理任何剩余的单独 ___ (可能是其他医生名字或日期)
    remaining_underscores = re.findall(r'\b___\b', text)
    for _ in remaining_underscores:
        # 根据上下文决定填充什么
        replacement = fake.last_name()
        text = re.sub(r'\b___\b', replacement, text, count=1)
        generator.add_annotation(replacement, "OTHER_PHI", "Other redacted information")

    return text, generator.get_annotations_json()


def process_csv_file(input_file, output_file, max_rows=None):
    """处理CSV文件，填充PHI并添加标注列"""
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Using LLM: {USE_LLM} (Model: {MODEL_NAME})")
    if max_rows:
        print(f"Test mode: Processing only first {max_rows} non-empty rows")

    rows_processed = 0

    # 尝试不同的编码方式
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    infile = None
    encoding_used = None
    
    for encoding in encodings:
        try:
            infile = open(input_file, 'r', encoding=encoding, errors='replace', newline='')
            # 尝试读取第一行来验证编码
            infile.seek(0)
            infile.readline()
            infile.seek(0)
            encoding_used = encoding
            print(f"Opened file with {encoding} encoding")
            break
        except Exception as e:
            if infile:
                infile.close()
            continue
    
    if infile is None:
        raise ValueError(f"Could not read file {input_file} with any supported encoding")

    # 输出使用UTF-8，保持格式一致
    try:
        with infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.DictReader(infile)

            # 创建新的列名：原有列 + filled_text + phi_annotations
            fieldnames = list(reader.fieldnames)
            if 'filled_text' not in fieldnames:
                fieldnames.append('filled_text')
            if 'phi_annotations' not in fieldnames:
                fieldnames.append('phi_annotations')
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for row in reader:
                extracted_text = row.get('extracted_text', '')
                original_col2 = row.get('original_col2', '')

                # 跳过空行或标题行
                if not extracted_text or extracted_text.strip() == '':
                    row['filled_text'] = extracted_text
                    row['phi_annotations'] = '[]'
                    writer.writerow(row)
                    continue

                # 如果设置了max_rows，检查是否已达到限制
                if max_rows and rows_processed >= max_rows:
                    break

                # 填充PHI
                filled_text, annotations = fill_phi_in_text(extracted_text, original_col2)

                # 更新行数据
                row['filled_text'] = filled_text
                row['phi_annotations'] = annotations

                writer.writerow(row)
                rows_processed += 1

                if rows_processed % 10 == 0:
                    print(f"Processed {rows_processed} rows...")

        print(f"Completed! Total rows processed: {rows_processed}")
        print(f"Output file saved to: {output_file}")
    finally:
        if infile:
            infile.close()


if __name__ == "__main__":
    input_csv = "patient_extracted.csv"
    output_csv = "patient_extracted_with_synth_phi_llm_refined.csv"
    MAX_ROWS = None  # Set to None to process all rows, or a number to limit (for testing)

    print("=" * 60)
    print("Enhanced PHI Data Filling Tool (with LLM support)")
    print("=" * 60)

    try:
        process_csv_file(input_csv, output_csv, max_rows=MAX_ROWS)
        print("\n" + "="*60)
        print("SUCCESS! Completed successfully!")
        print("="*60)
        print(f"\nScript location: {__file__}")
        print(f"Output file: {output_csv}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

