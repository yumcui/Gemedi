#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI Bad Data Filling Tool - Generate Low-Quality Training Data

Intentionally introduces issues in three dimensions:
1. Linguistic Realism - Violates language rules
2. Clinical Consistency - Clinical inconsistencies
3. Document Structure - Destroys document structure

Used to generate negative samples for training data
"""

import csv
import json
import re
from datetime import datetime, timedelta, date
from faker import Faker
import random

# Initialize Faker generator
fake = Faker()
Faker.seed(123)  # Different seed to avoid overlap with good data
random.seed(123)


class BadDataType:
    """Bad data types"""
    LINGUISTIC_MILD = "linguistic_mild"        # Mild language issues
    LINGUISTIC_SEVERE = "linguistic_severe"    # Severe language issues
    CLINICAL_AGE = "clinical_age"              # Age-diagnosis mismatch
    CLINICAL_SEX = "clinical_sex"              # Sex-diagnosis mismatch
    STRUCTURE_MILD = "structure_mild"          # Mild structure issues
    STRUCTURE_SEVERE = "structure_severe"      # Severe structure issues
    MIXED_BAD = "mixed_bad"                    # Mixed multiple issues


class BadPHIGenerator:
    """Generate intentionally problematic PHI data"""

    def __init__(self, bad_data_type=None):
        self.current_patient = {}
        self.phi_annotations = []
        self.bad_phi_details = []  # Record which PHI fields have issues
        self.bad_data_type = bad_data_type or random.choice([
            BadDataType.LINGUISTIC_MILD,
            BadDataType.LINGUISTIC_SEVERE,
            BadDataType.CLINICAL_AGE,
            BadDataType.CLINICAL_SEX,
            BadDataType.STRUCTURE_MILD,
            BadDataType.STRUCTURE_SEVERE,
            BadDataType.MIXED_BAD
        ])

    def reset_patient(self, bad_data_type=None):
        """Reset data for a new patient"""
        self.current_patient = {}
        self.phi_annotations = []
        self.bad_phi_details = []  # Reset bad data details
        if bad_data_type:
            self.bad_data_type = bad_data_type
        else:
            # Randomly select bad data type
            self.bad_data_type = random.choice([
                BadDataType.LINGUISTIC_MILD,
                BadDataType.LINGUISTIC_MILD,  # Increase weight
                BadDataType.LINGUISTIC_SEVERE,
                BadDataType.CLINICAL_AGE,
                BadDataType.CLINICAL_SEX,
                BadDataType.STRUCTURE_MILD,
                BadDataType.STRUCTURE_SEVERE,
                BadDataType.MIXED_BAD
            ])
    
    def add_bad_phi_detail(self, field_name, value, issue_type, description):
        """Record problematic PHI fields"""
        self.bad_phi_details.append({
            "field": field_name,
            "value": value,
            "issue_type": issue_type,
            "description": description
        })
    
    def get_bad_phi_details_json(self):
        """Return bad data details in JSON format"""
        return json.dumps(self.bad_phi_details, ensure_ascii=False)

    def should_apply_linguistic_issues(self):
        """Whether to apply linguistic issues
        Note: Now always returns True to ensure each bad data has at least one linguistic issue
        """
        return True  # Ensure each bad data has at least one linguistic issue

    def should_apply_clinical_issues(self):
        """Whether to apply clinical issues (random decision)"""
        if self.bad_data_type in [
            BadDataType.CLINICAL_AGE,
            BadDataType.CLINICAL_SEX,
            BadDataType.MIXED_BAD
        ]:
            return True
        # For other types, randomly decide whether to apply clinical issues (30% probability)
        return random.random() < 0.3

    def should_apply_structure_issues(self):
        """Whether to apply structure issues (random decision)"""
        if self.bad_data_type in [
            BadDataType.STRUCTURE_MILD,
            BadDataType.STRUCTURE_SEVERE,
            BadDataType.MIXED_BAD
        ]:
            return True
        # For other types, randomly decide whether to apply structure issues (30% probability)
        return random.random() < 0.3

    def generate_bad_patient_name(self, sex=None):
        """Generate problematic patient name
        Ensures at least one linguistic issue (detectable by preprocess)
        """
        # Always generate problematic name (because should_apply_linguistic_issues always returns True)
        bad_name_types = []
        
        # Determine issue type based on severity
        if self.bad_data_type == BadDataType.LINGUISTIC_SEVERE:
            bad_name_types = [
                'single_word',      # Single word - will be detected as "less than 2 words"
                'weird_chars',      # Weird characters - will be detected as "contains special characters"
                'no_capitalization', # No capitalization - will be detected as "first letter not capitalized"
                'single_letter',     # Single letter - will be detected as "single letter"
                'numbers'           # Contains numbers - will be detected as "contains special characters"
            ]
        else:
            # Mild issues: Select issues that will be detected by preprocess
            bad_name_types = [
                'no_capitalization',  # No capitalization - will be detected as "first letter not capitalized"
                'single_word'         # Only single word - will be detected as "less than 2 words"
            ]
        
        issue_type = random.choice(bad_name_types)
        
        if issue_type == 'single_word':
            # Only last name, no first name - will be detected by preprocess as "less than 2 words"
            full_name = fake.last_name()
            first_name = full_name
            last_name = ""
            self.add_bad_phi_detail("NAME", full_name, issue_type, "Name has only one word (missing first or last name)")
        elif issue_type == 'weird_chars':
            # Contains special characters - will be detected by preprocess as "contains special characters"
            first_name = fake.first_name()
            last_name = fake.last_name()
            weird_chars = ['@', '#', '$', '123', '!!!', '???', '___']
            char = random.choice(weird_chars)
            full_name = f"{first_name}{char} {last_name}"
            self.add_bad_phi_detail("NAME", full_name, issue_type, f"Name contains weird characters: {char}")
        elif issue_type == 'no_capitalization':
            # All lowercase - will be detected by preprocess as "first letter not capitalized"
            first_name = fake.first_name().lower()
            last_name = fake.last_name().lower()
            full_name = f"{first_name} {last_name}"
            self.add_bad_phi_detail("NAME", full_name, issue_type, "Name not properly capitalized")
        elif issue_type == 'single_letter':
            # Single letter - will be detected by preprocess as "single letter"
            full_name = random.choice(['A', 'B', 'X', 'Z'])
            first_name = full_name
            last_name = ""
            self.add_bad_phi_detail("NAME", full_name, issue_type, "Name is a single letter")
        elif issue_type == 'numbers':
            # Contains numbers - will be detected by preprocess as "contains special characters"
            first_name = fake.first_name()
            last_name = fake.last_name()
            num = random.randint(1, 99)
            full_name = f"{first_name}{num} {last_name}"
            self.add_bad_phi_detail("NAME", full_name, issue_type, f"Name contains numbers: {num}")
        else:
            # Default: single word (ensure there's an issue)
            full_name = fake.last_name()
            first_name = full_name
            last_name = ""
            self.add_bad_phi_detail("NAME", full_name, "single_word", "Name has only one word (default)")

        self.current_patient['name'] = full_name
        self.current_patient['first_name'] = first_name if 'first_name' in locals() else full_name
        self.current_patient['last_name'] = last_name if 'last_name' in locals() else ""
        return full_name

    def generate_unit_no(self):
        """Generate unit number"""
        unit_no = str(random.randint(1000000, 9999999))
        self.current_patient['unit_no'] = unit_no
        return unit_no

    def generate_bad_dates(self):
        """Generate problematic dates
        Ensures at least one linguistic issue (detectable by preprocess)
        """
        # Always generate problematic dates (because should_apply_linguistic_issues always returns True)
        # Problematic date types (detectable by preprocess)
        bad_date_types = [
            'future_date',           # Future date - will be detected as "date is in the future"
            'too_old',               # More than 105 years ago - will be detected as "date too old"
            'reversed',              # Discharge before admission - will be detected as "admission date >= discharge date"
            'same_date',             # Same day - will be detected as "admission date >= discharge date"
        ]
        
        # Select issue type based on bad_data_type
        if self.bad_data_type == BadDataType.LINGUISTIC_SEVERE:
            # Severe issues: future date or date too old
            issue_type = random.choice(['future_date', 'too_old', 'reversed'])
        else:
            # Mild issues: select issues that will be detected
            issue_type = random.choice(['future_date', 'reversed', 'same_date'])
        
        if issue_type == 'future_date':
                # Future date
                admission_date = fake.date_between(start_date='+1d', end_date='+1y')
                discharge_date = admission_date + timedelta(days=random.randint(1, 30))
                self.add_bad_phi_detail("ADMISSION_DATE", admission_date.strftime('%Y-%m-%d'), issue_type, "Admission date is in the future")
                self.add_bad_phi_detail("DISCHARGE_DATE", discharge_date.strftime('%Y-%m-%d'), issue_type, "Discharge date is in the future")
        elif issue_type == 'too_old':
                # More than 105 years ago
                admission_date = fake.date_between(start_date='-120y', end_date='-106y')
                discharge_date = admission_date + timedelta(days=random.randint(1, 30))
                self.add_bad_phi_detail("ADMISSION_DATE", admission_date.strftime('%Y-%m-%d'), issue_type, f"Admission date is too old: {admission_date.year}")
                self.add_bad_phi_detail("DISCHARGE_DATE", discharge_date.strftime('%Y-%m-%d'), issue_type, f"Discharge date is too old: {discharge_date.year}")
        elif issue_type == 'reversed':
                # Discharge date before admission date
                discharge_date = fake.date_between(start_date='-2y', end_date='-30d')
                admission_date = discharge_date + timedelta(days=random.randint(1, 30))
                self.add_bad_phi_detail("ADMISSION_DATE", admission_date.strftime('%Y-%m-%d'), issue_type, f"Admission date ({admission_date}) is after discharge date ({discharge_date})")
                self.add_bad_phi_detail("DISCHARGE_DATE", discharge_date.strftime('%Y-%m-%d'), issue_type, f"Discharge date ({discharge_date}) is before admission date ({admission_date})")
        elif issue_type == 'same_date':
                # Same day admission and discharge
                admission_date = fake.date_between(start_date='-2y', end_date='-30d')
                discharge_date = admission_date
                self.add_bad_phi_detail("ADMISSION_DATE", admission_date.strftime('%Y-%m-%d'), issue_type, "Admission and discharge are on the same day")
                self.add_bad_phi_detail("DISCHARGE_DATE", discharge_date.strftime('%Y-%m-%d'), issue_type, "Admission and discharge are on the same day")
        else:
                admission_date = fake.date_between(start_date='-2y', end_date='-30d')
                discharge_date = admission_date + timedelta(days=random.randint(1, 30))

        # Format dates
        admission_str = admission_date.strftime('%Y-%m-%d')
        discharge_str = discharge_date.strftime('%Y-%m-%d')

        self.current_patient['admission_date'] = admission_str
        self.current_patient['discharge_date'] = discharge_str

        return admission_str, discharge_str

    def generate_bad_dob(self, sex=None):
        """Generate date of birth"""
        if self.should_apply_clinical_issues() and self.bad_data_type == BadDataType.CLINICAL_AGE:
            # Generate extreme age for clinical inconsistency
            age_type = random.choice(['too_young', 'too_old'])
            if age_type == 'too_young':
                # Too young (for geriatric diseases)
                dob = fake.date_of_birth(minimum_age=5, maximum_age=25)
                age = self._calculate_age(dob)
                self.add_bad_phi_detail("DATE_OF_BIRTH", dob.strftime('%Y-%m-%d'), age_type, f"Age ({age} years) is too young for typical diagnosis")
            else:
                # Too old (for young people's diseases)
                dob = fake.date_of_birth(minimum_age=85, maximum_age=100)
                age = self._calculate_age(dob)
                self.add_bad_phi_detail("DATE_OF_BIRTH", dob.strftime('%Y-%m-%d'), age_type, f"Age ({age} years) is too old for typical diagnosis")
        else:
            # Normal age range
            dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        
        dob_str = dob.strftime('%Y-%m-%d')
        self.current_patient['dob'] = dob_str
        self.current_patient['age_at_discharge'] = self._calculate_age(dob)
        return dob_str

    def _calculate_age(self, dob):
        """Calculate age"""
        if isinstance(dob, str):
            dob = datetime.strptime(dob, '%Y-%m-%d').date()
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age

    def generate_bad_attending_name(self):
        """Generate attending physician name (may have issues)
        Ensures at least one linguistic issue (detectable by preprocess)
        """
        # Decide whether to apply issues based on severity
        if self.bad_data_type == BadDataType.LINGUISTIC_SEVERE:
            # Severe issues: missing Dr. prefix or format error - will be detected as "doctor name format error"
            bad_formats = [
                (f"{fake.first_name()} {fake.last_name()}", "missing_dr", "Attending name missing 'Dr.' prefix"),
                (f"Doctor {fake.last_name()}", "wrong_prefix", "Attending uses 'Doctor' instead of 'Dr.'"),
                (f"dr {fake.last_name()}", "lowercase_dr", "Attending has lowercase 'dr' instead of 'Dr.'"),
            ]
            attending, issue_type, description = random.choice(bad_formats)
            self.add_bad_phi_detail("ATTENDING_PROVIDER", attending, issue_type, description)
        elif random.random() < 0.3:  # 30% probability to apply mild issues
            # Mild issues: format error
            bad_formats = [
                (f"dr {fake.last_name()}", "lowercase_dr", "Attending has lowercase 'dr' instead of 'Dr.'"),
            ]
            attending, issue_type, description = random.choice(bad_formats)
            self.add_bad_phi_detail("ATTENDING_PROVIDER", attending, issue_type, description)
        else:
            # Normal format (but name may have issues, handled by generate_bad_patient_name)
            attending = f"Dr. {fake.first_name()} {fake.last_name()}"
        
        self.current_patient['attending'] = attending
        return attending

    def generate_facility_name(self):
        """Generate healthcare facility name"""
        facilities = [
            "Home Health Services", "Visiting Nurse Association",
            "Community Health Services", "Home Care Partners",
            "Premier Home Health", "Compassionate Care Services"
        ]
        facility = random.choice(facilities)
        self.current_patient['facility'] = facility
        return facility

    def generate_hospital_name(self):
        """Generate hospital name"""
        hospitals = [
            "Metropolitan General Hospital", "City Medical Center",
            "Regional Health Center", "University Hospital",
            "Community Memorial Hospital", "St. Mary's Medical Center"
        ]
        hospital = random.choice(hospitals)
        self.current_patient['hospital'] = hospital
        return hospital

    def add_annotation(self, value, phi_type, reason):
        """Add PHI annotation"""
        self.phi_annotations.append({
            "value": value,
            "type": phi_type,
            "reason": reason
        })

    def get_annotations_json(self):
        """Return annotations in JSON format"""
        return json.dumps(self.phi_annotations, ensure_ascii=False)


def apply_structure_damage(text, generator):
    """
    Damage document structure
    Can apply multiple structure issues (shuffle order, remove Attending, remove keywords, etc.)
    """
    if not generator.should_apply_structure_issues():
        return text
    
    lines = text.split('\n')
    structure_issues = []
    all_keywords = [
        'Discharge Disposition:',
        'Discharge Diagnosis:',
        'Discharge Condition:',
        'Discharge Instructions:',
        'Followup Instructions:'
    ]
    
    if generator.bad_data_type == BadDataType.STRUCTURE_SEVERE or generator.bad_data_type == BadDataType.MIXED_BAD:
        # Severely damage structure: apply multiple issues
        
        # 1. Shuffle order of first 3 lines (affects check_structure_order)
        if len(lines) >= 3:
            non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
            if len(non_empty_indices) >= 3:
                # Find indices of first 3 non-empty lines
                first_three_indices = non_empty_indices[:3]
                first_three_lines = [lines[i] for i in first_three_indices]
                random.shuffle(first_three_lines)
                # Reassign
                for idx, new_line in zip(first_three_indices, first_three_lines):
                    lines[idx] = new_line
                structure_issues.append("First 3 lines order shuffled")
        
        # 2. Remove Attending field (affects check_attending)
        text = '\n'.join(lines)
        if 'Attending:' in text:
            text = re.sub(r'Attending:.*?\n', '', text)
            structure_issues.append("Attending field removed")
        
        # 3. Remove multiple keywords (affects check_keywords)
        # Remove 2-4 keywords
        num_to_remove = random.randint(2, 4)
        keywords_to_remove = random.sample(all_keywords, k=num_to_remove)
        
        for keyword in keywords_to_remove:
            # Remove entire section
            pattern = f'{re.escape(keyword)}.*?(?=\\n\\n|\\n[A-Z][a-z]+:|$)'
            if re.search(pattern, text, flags=re.DOTALL):
                text = re.sub(pattern, '', text, flags=re.DOTALL)
                structure_issues.append(f"Missing keyword: {keyword}")
    
    elif generator.bad_data_type == BadDataType.STRUCTURE_MILD:
        # Mildly damage structure: apply 1-3 issues
        
        # Decide which issues to apply (can be multiple)
        issues_to_apply = []
        
        # 30% probability to shuffle first 3 lines
        if random.random() < 0.3:
            issues_to_apply.append('shuffle_lines')
        
        # 50% probability to remove Attending
        if random.random() < 0.5:
            issues_to_apply.append('remove_attending')
        
        # 70% probability to remove 1-2 keywords
        if random.random() < 0.7:
            issues_to_apply.append('remove_keywords')
        
        # If no issues selected, apply at least one
        if not issues_to_apply:
            issues_to_apply.append(random.choice(['remove_attending', 'remove_keywords']))
        
        # Apply selected issues
        text = '\n'.join(lines)
        
        if 'shuffle_lines' in issues_to_apply:
            # Shuffle first 3 lines
            non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
            if len(non_empty_indices) >= 3:
                first_three_indices = non_empty_indices[:3]
                first_three_lines = [lines[i] for i in first_three_indices]
                random.shuffle(first_three_lines)
                for idx, new_line in zip(first_three_indices, first_three_lines):
                    lines[idx] = new_line
                text = '\n'.join(lines)
                structure_issues.append("First 3 lines order shuffled")
        
        if 'remove_attending' in issues_to_apply:
            # Remove Attending
            if 'Attending:' in text:
                text = re.sub(r'Attending:.*?\n', '', text)
                structure_issues.append("Attending field removed")
        
        if 'remove_keywords' in issues_to_apply:
            # Remove 1-2 keywords
            num_to_remove = random.randint(1, 2)
            keywords_to_remove = random.sample(all_keywords, k=min(num_to_remove, len(all_keywords)))
            
            for keyword in keywords_to_remove:
                pattern = f'{re.escape(keyword)}.*?(?=\\n\\n|\\n[A-Z][a-z]+:|$)'
                if re.search(pattern, text, flags=re.DOTALL):
                    text = re.sub(pattern, '', text, flags=re.DOTALL)
                    structure_issues.append(f"Missing keyword: {keyword}")
    
    else:
        # Other types (e.g., CLINICAL_AGE, CLINICAL_SEX, etc.): randomly apply 1-2 structure issues
        
        # Decide which issues to apply
        issues_to_apply = []
        
        # 20% probability to shuffle first 3 lines
        if random.random() < 0.2:
            issues_to_apply.append('shuffle_lines')
        
        # 30% probability to remove Attending
        if random.random() < 0.3:
            issues_to_apply.append('remove_attending')
        
        # 40% probability to remove 1-2 keywords
        if random.random() < 0.4:
            issues_to_apply.append('remove_keywords')
        
        # Apply selected issues
        text = '\n'.join(lines)
        
        if 'shuffle_lines' in issues_to_apply:
            non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
            if len(non_empty_indices) >= 3:
                first_three_indices = non_empty_indices[:3]
                first_three_lines = [lines[i] for i in first_three_indices]
                random.shuffle(first_three_lines)
                for idx, new_line in zip(first_three_indices, first_three_lines):
                    lines[idx] = new_line
                text = '\n'.join(lines)
                structure_issues.append("First 3 lines order shuffled")
        
        if 'remove_attending' in issues_to_apply:
            if 'Attending:' in text:
                text = re.sub(r'Attending:.*?\n', '', text)
                structure_issues.append("Attending field removed")
        
        if 'remove_keywords' in issues_to_apply:
            num_to_remove = random.randint(1, 2)
            keywords_to_remove = random.sample(all_keywords, k=min(num_to_remove, len(all_keywords)))
            
            for keyword in keywords_to_remove:
                pattern = f'{re.escape(keyword)}.*?(?=\\n\\n|\\n[A-Z][a-z]+:|$)'
                if re.search(pattern, text, flags=re.DOTALL):
                    text = re.sub(pattern, '', text, flags=re.DOTALL)
                    structure_issues.append(f"Missing keyword: {keyword}")
    
    # Record structure issues
    if structure_issues:
        for issue in structure_issues:
            generator.add_bad_phi_detail("DOCUMENT_STRUCTURE", "", "structure_issue", issue)
    
    return text


def fill_phi_in_text_bad(text, original_col2):
    """Fill all PHI placeholders in text (intentionally generate bad data)"""
    generator = BadPHIGenerator()

    # First extract sex information (if available)
    sex = None
    sex_match = re.search(r'Sex:\s*([MF])', text)
    original_sex = None
    if sex_match:
        sex = sex_match.group(1)
        original_sex = sex
    
    # For sex inconsistency cases, intentionally use wrong sex
    if generator.bad_data_type == BadDataType.CLINICAL_SEX:
        # Reverse sex for name generation
        reversed_sex = 'M' if sex == 'F' else 'F' if sex == 'M' else sex
        if reversed_sex != sex:
            generator.add_bad_phi_detail("SEX", reversed_sex, "sex_mismatch", 
                                       f"Sex mismatch: Text indicates '{sex}' but name suggests '{reversed_sex}'")
        sex = reversed_sex

    # Process PHI fields in order

    # 1. Name (patient name)
    name_pattern = r'Name:\s*___'
    if re.search(name_pattern, text):
        patient_name = generator.generate_bad_patient_name(sex)
        text = re.sub(name_pattern, f'Name:  {patient_name}', text, count=1)
        generator.add_annotation(patient_name, "NAME", "Patient's name")

    # 2. Unit No (medical record number)
    unit_pattern = r'Unit No:\s*___'
    if re.search(unit_pattern, text):
        unit_no = generator.generate_unit_no()
        text = re.sub(unit_pattern, f'Unit No:   {unit_no}', text, count=1)
        generator.add_annotation(unit_no, "UNIT_NO", "Patient's unit number")

    # 3. Admission Date and Discharge Date (admission and discharge dates)
    admission_pattern = r'Admission Date:\s*___'
    discharge_pattern = r'Discharge Date:\s*___'
    if re.search(admission_pattern, text) or re.search(discharge_pattern, text):
        admission_date, discharge_date = generator.generate_bad_dates()
        text = re.sub(admission_pattern, f'Admission Date:  {admission_date}', text)
        text = re.sub(discharge_pattern, f'Discharge Date:   {discharge_date}', text)

        if re.search(r'Admission Date:', text):
            generator.add_annotation(admission_date, "DATE_OF_ADMISSION", "Admission date")
        if re.search(r'Discharge Date:', text):
            generator.add_annotation(discharge_date, "DISCHARGE_DATE", "Discharge date")

    # 4. Date of Birth (date of birth)
    dob_pattern = r'Date of Birth:\s*___'
    if re.search(dob_pattern, text):
        dob = generator.generate_bad_dob(sex)
        text = re.sub(dob_pattern, f'Date of Birth:  {dob}', text, count=1)
        generator.add_annotation(dob, "DATE_OF_BIRTH", "Patient's date of birth")

    # 5. Attending (attending physician)
    attending_pattern = r'Attending:\s*___\.?'
    if re.search(attending_pattern, text):
        attending = generator.generate_bad_attending_name()
        text = re.sub(attending_pattern, f'Attending: {attending}', text)
        generator.add_annotation(attending, "ATTENDING_PROVIDER", "Attending physician's name")

    # 6. Dear Mr./Ms. ___ (name in salutation)
    dear_pattern = r'Dear (Mr\.|Ms\.)\s*___,'
    dear_match = re.search(dear_pattern, text)
    if dear_match:
        title = dear_match.group(1)
        if generator.current_patient.get('last_name'):
            last_name = generator.current_patient['last_name']
        else:
            if title == 'Ms.':
                full_name = generator.generate_bad_patient_name('F')
            else:
                full_name = generator.generate_bad_patient_name('M')
            last_name = generator.current_patient['last_name']

        text = re.sub(dear_pattern, f'Dear {title} {last_name},', text)
        generator.add_annotation(title, "NAME_TITLE", "Name title")
        generator.add_annotation(last_name, "DISCHARGE_INSTRUCTIONS_NAME", "Name in discharge instructions")

    # 7. Dr. ___ (other doctor references)
    dr_pattern = r'Dr\.\s*___'
    dr_matches = list(re.finditer(dr_pattern, text))
    for match in dr_matches:
        if generator.current_patient.get('attending'):
            doctor_name = generator.current_patient['attending']
        else:
            doctor_name = generator.generate_bad_attending_name()
        text = text[:match.start()] + doctor_name + text[match.end():]
        generator.add_annotation(doctor_name, "PROVIDER_NAME", "Provider name in text")

    # 8. Your ___ Team (hospital/team name)
    team_pattern = r'Your\s+___\s+Team'
    if re.search(team_pattern, text):
        hospital = generator.generate_hospital_name()
        text = re.sub(team_pattern, f'Your {hospital} Team', text)
        generator.add_annotation(hospital, "HOSPITAL_NAME", "Hospital name")

    # 9. Facility: ___ (healthcare facility)
    facility_pattern = r'Facility:\s*___'
    if re.search(facility_pattern, text):
        facility = generator.generate_facility_name()
        text = re.sub(facility_pattern, f'Facility:\n{facility}', text)
        generator.add_annotation(facility, "FACILITY_NAME", "Healthcare facility name")

    # 10. ___ at ___ (hospital name reference)
    at_hospital_pattern = r'at\s+___[\.\s]'
    if re.search(at_hospital_pattern, text):
        hospital = generator.generate_hospital_name()
        text = re.sub(at_hospital_pattern, f'at {hospital}. ', text)
        generator.add_annotation(hospital, "HOSPITAL_NAME", "Hospital name reference")

    # 11. Followup Instructions: ___ (follow-up instructions)
    followup_pattern = r'Followup Instructions:\s*___'
    if re.search(followup_pattern, text):
        followup_text = f"Follow up with {generator.current_patient.get('attending', 'Dr. ' + fake.last_name())} in 2 weeks"
        text = re.sub(followup_pattern, f'Followup Instructions:\n{followup_text}', text)
        generator.add_annotation(followup_text, "FOLLOWUP_INSTRUCTIONS", "Follow-up instructions")

    # 12. Handle any remaining standalone ___
    remaining_underscores = re.findall(r'\b___\b', text)
    for _ in remaining_underscores:
        replacement = fake.last_name()
        text = re.sub(r'\b___\b', replacement, text, count=1)
        generator.add_annotation(replacement, "OTHER_PHI", "Other redacted information")

    # Apply document structure damage
    text = apply_structure_damage(text, generator)

    # Get bad data details
    bad_phi_details_json = generator.get_bad_phi_details_json()

    return text, generator.get_annotations_json(), generator.bad_data_type, bad_phi_details_json


def process_csv_file(input_file, output_file):
    """Process CSV file, fill PHI and add annotation columns (bad data version)"""
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    print("Generating BAD DATA for training...")

    rows_processed = 0
    bad_data_stats = {}

    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    infile = None
    encoding_used = None
    
    for encoding in encodings:
        try:
            infile = open(input_file, 'r', encoding=encoding, errors='replace', newline='')
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

    try:
        with infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.DictReader(infile)

            # Create new column names
            fieldnames = list(reader.fieldnames)
            if 'filled_text' not in fieldnames:
                fieldnames.append('filled_text')
            if 'phi_annotations' not in fieldnames:
                fieldnames.append('phi_annotations')
            if 'bad_data_type' not in fieldnames:
                fieldnames.append('bad_data_type')
            if 'bad_phi_details' not in fieldnames:
                fieldnames.append('bad_phi_details')
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for row in reader:
                extracted_text = row.get('extracted_text', '')
                original_col2 = row.get('original_col2', '')

                # Skip empty rows
                if not extracted_text or extracted_text.strip() == '':
                    row['filled_text'] = extracted_text
                    row['phi_annotations'] = '[]'
                    row['bad_data_type'] = 'none'
                    row['bad_phi_details'] = '[]'
                    writer.writerow(row)
                    continue

                # Fill PHI (bad data version)
                filled_text, annotations, bad_type, bad_phi_details = fill_phi_in_text_bad(extracted_text, original_col2)

                # Count bad data types
                bad_data_stats[bad_type] = bad_data_stats.get(bad_type, 0) + 1

                # Update row data
                row['filled_text'] = filled_text
                row['phi_annotations'] = annotations
                row['bad_data_type'] = bad_type
                row['bad_phi_details'] = bad_phi_details

                writer.writerow(row)
                rows_processed += 1

                if rows_processed % 100 == 0:
                    print(f"Processed {rows_processed} rows...")

        print(f"\nCompleted! Total rows processed: {rows_processed}")
        print(f"Output file saved to: {output_file}")
        
        print("\n" + "="*60)
        print("Bad Data Type Statistics:")
        print("="*60)
        for bad_type, count in sorted(bad_data_stats.items()):
            percentage = (count / rows_processed * 100) if rows_processed > 0 else 0
            print(f"{bad_type:30s}: {count:5d} ({percentage:5.1f}%)")
        print("="*60)
        print("\nNote: Check 'bad_phi_details' column for specific issues in each row")
        print("="*60)
    finally:
        if infile:
            infile.close()


if __name__ == "__main__":
    input_csv = "patient_extracted.csv"
    output_csv = "patient_extracted_with_bad_phi.csv"

    print("=" * 60)
    print("PHI Bad Data Filling Tool")
    print("Generating Low-Quality Training Data")
    print("=" * 60)
    print("\nBad Data Types:")
    print("  - linguistic_mild: Mild language issues")
    print("  - linguistic_severe: Severe language issues")
    print("  - clinical_age: Age-diagnosis mismatch")
    print("  - clinical_sex: Sex-diagnosis mismatch")
    print("  - structure_mild: Mild structure issues")
    print("  - structure_severe: Severe structure issues")
    print("  - mixed_bad: Mixed multiple issues")
    print("=" * 60)

    try:
        process_csv_file(input_csv, output_csv)
        print("\n" + "="*60)
        print("SUCCESS! Bad data generation completed!")
        print("="*60)
        print(f"\nOutput file: {output_csv}")
        print("\nNote: This file contains intentionally generated low-quality data for training models to identify problematic data.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

