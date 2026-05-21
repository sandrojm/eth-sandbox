"""Clinical note extraction functions for vitals, labs, and medications."""

import re

# =============================================================================
# VITAL SIGNS PATTERNS AND CODES
# =============================================================================

VITAL_PATTERNS = {
    'bp': r'(?:BP|Blood Pressure)[:\s]*(\d{2,3})/(\d{2,3})',
    'weight_kg': r'(?:Weight|Wt)[:\s]*(\d+\.?\d*)\s*(?:kg|KG|Kg)',
    'weight_lb': r'(?:Weight|Wt)[:\s]*(\d+\.?\d*)\s*(?:lb|lbs|LB)',
    'height_cm': r'(?:Height|Ht)[:\s]*(\d+\.?\d*)\s*(?:cm|CM)',
    'spo2': r'(?:SpO2|O2 Sat|Oxygen)[:\s]*(\d{2,3})\s*%',
    'temp': r'(?i)(?:Temp|Temperature|T)[:\s]*(\d{2,3}\.?\d*)\s*(?:F|C|Fahrenheit|Celsius|°F|°C)'
}

VITAL_LOINC_CODES = {
    'Systolic Blood Pressure': '8480-6',
    'Diastolic Blood Pressure': '8462-4',
    'Body Weight': '29463-7',
    'Body Temperature': '8310-5',
    'Oxygen saturation': '2708-6'
}

# =============================================================================
# LAB VALUE PATTERNS AND CODES
# =============================================================================

LAB_PATTERNS = {
    'hba1c': r'(?:HbA1c|A1c|Hemoglobin A1c)[:\s]*(\d+\.?\d*)\s*%?',
    'glucose': r'(?:Glucose|Blood Sugar|BG)[:\s]*(\d+)\s*(?:mg/dL)?',
    'creatinine': r'(?:Creatinine|Cr)[:\s]*(\d+\.?\d*)\s*(?:mg/dL)?',
    'egfr': r'(?:eGFR|GFR)[:\s]*(\d+)\s*(?:mL/min)?'
}

LAB_LOINC_CODES = {
    'hba1c': ('4548-4', 'Hemoglobin A1c', '%'),
    'glucose': ('2339-0', 'Glucose', 'mg/dL'),
    'creatinine': ('2160-0', 'Creatinine', 'mg/dL'),
    'egfr': ('33914-3', 'eGFR', 'mL/min')
}

LAB_RANGES = {
    'hba1c': (4.0, 15.0),
    'glucose': (50, 500),
    'creatinine': (0.3, 10.0),
    'egfr': (5, 150)
}

# =============================================================================
# MEDICATIONS
# =============================================================================

KNOWN_MEDICATIONS = {
    # --- DIABETES: Biguanides ---
    'metformin': 'Metformin',
    'glucophage': 'Metformin (Glucophage)',

    # --- DIABETES: Sulfonylureas (High hypoglycemia risk) ---
    'glipizide': 'Glipizide',
    'glyburide': 'Glyburide',
    'glimepiride': 'Glimepiride',
    'amaryl': 'Glimepiride (Amaryl)',

    # --- DIABETES: SGLT2 Inhibitors (Cardio/Renal Protective) ---
    'jardiance': 'Empagliflozin (Jardiance)',
    'farxiga': 'Dapagliflozin (Farxiga)',
    'invokana': 'Canagliflozin (Invokana)',
    'steglatro': 'Ertugliflozin (Steglatro)',

    # --- DIABETES: GLP-1 Receptor Agonists (Weight/Cardio Protective) ---
    'ozempic': 'Semaglutide (Ozempic)',
    'wegovy': 'Semaglutide (Wegovy)',
    'rybelsus': 'Semaglutide (Rybelsus)',
    'trulicity': 'Dulaglutide (Trulicity)',
    'victoza': 'Liraglutide (Victoza)',
    'mounjaro': 'Tirzepatide (Mounjaro)',

    # --- DIABETES: DPP-4 Inhibitors ---
    'januvia': 'Sitagliptin (Januvia)',
    'onglyza': 'Saxagliptin (Onglyza)',
    'tradjenta': 'Linagliptin (Tradjenta)',

    # --- DIABETES: Insulins ---
    'insulin': 'Insulin',
    'lantus': 'Insulin Glargine (Lantus)',
    'basaglar': 'Insulin Glargine (Basaglar)',
    'toujeo': 'Insulin Glargine (Toujeo)',
    'humalog': 'Insulin Lispro (Humalog)',
    'novolog': 'Insulin Aspart (Novolog)',
    'fiasp': 'Insulin Aspart (Fiasp)',
    'levemir': 'Insulin Detemir (Levemir)',
    'tresiba': 'Insulin Degludec (Tresiba)',

    # --- CARDIOVASCULAR: ACE Inhibitors & ARBs (Kidney Protection) ---
    'lisinopril': 'Lisinopril',
    'zestril': 'Lisinopril (Zestril)',
    'losartan': 'Losartan (Cozaar)',
    'valsartan': 'Valsartan (Diovan)',
    'enalapril': 'Enalapril (Vasotec)',

    # --- CARDIOVASCULAR: Statins (Cholesterol) ---
    'atorvastatin': 'Atorvastatin (Lipitor)',
    'simvastatin': 'Simvastatin (Zocor)',
    'rosuvastatin': 'Rosuvastatin (Crestor)',
    'pravastatin': 'Pravastatin (Pravachol)',

    # --- CARDIOVASCULAR: Blood Pressure & Heart Health ---
    'amlodipine': 'Amlodipine (Norvasc)',
    'metoprolol': 'Metoprolol (Lopressor)',
    'carvedilol': 'Carvedilol (Coreg)',
    'furosemide': 'Furosemide (Lasix)',
    'aspirin': 'Aspirin',
    'clopidogrel': 'Clopidogrel (Plavix)',

    # --- OTHER: Common Comorbidities ---
    'omeprazole': 'Omeprazole (Prilosec)',
    'gabapentin': 'Gabapentin (Neurontin)', # Often for diabetic neuropathy
    'metformin': 'Metformin',
}

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_vitals_from_note(note_text, patient_id, encounter_id):
    """
    Extract vital signs from clinical note text using VITAL_PATTERNS.

    Args:
        note_text: The clinical note text to extract from
        patient_id: Patient ID to include in extracted records
        encounter_id: Encounter ID to include in extracted records

    Returns:
        List of dicts, each containing a vital sign record with:
        PATIENT, ENCOUNTER, CODE, DESCRIPTION, VALUE, UNITS
    """
    vitals = []
    if not isinstance(note_text, str):
        return vitals

    # --- Blood Pressure ---
    # Blood pressure extraction creates TWO records: systolic and diastolic
    # TODO: Search for BP pattern using re.search() with VITAL_PATTERNS['bp']
    bp_match = re.search(VITAL_PATTERNS['bp'], note_text)

    if bp_match:
        # Systolic Blood Pressure (first captured group)
        # TODO: Create dict with PATIENT, ENCOUNTER, CODE, DESCRIPTION, VALUE, UNITS
        vitals.append({
            'PATIENT': patient_id,
            'ENCOUNTER': encounter_id,
            'CODE': VITAL_LOINC_CODES['Systolic Blood Pressure'],
            'DESCRIPTION': 'Systolic Blood Pressure',
            'VALUE': int(bp_match.group(1)),
            'UNITS': 'mmHg'
        })

        # Diastolic Blood Pressure (second captured group)
        # TODO: Create dict for diastolic
        vitals.append({
            'PATIENT': patient_id,
            'ENCOUNTER': encounter_id,
            'CODE': VITAL_LOINC_CODES['Diastolic Blood Pressure'],
            'DESCRIPTION': 'Diastolic Blood Pressure',
            'VALUE': int(bp_match.group(2)),
            'UNITS': 'mmHg'
        })

    # --- Weight ---
    # Check for kg first, then lbs (convert lbs to kg)
    # TODO: Search for weight patterns
    weight_kg = re.search(VITAL_PATTERNS['weight_kg'], note_text)
    weight_lb = re.search(VITAL_PATTERNS['weight_lb'], note_text)

    if weight_kg:
        vitals.append({
            'PATIENT': patient_id,
            'ENCOUNTER': encounter_id,
            'CODE': VITAL_LOINC_CODES['Body Weight'],
            'DESCRIPTION': 'Body Weight',
            'VALUE': float(weight_kg.group(1)),
            'UNITS': 'kg'
        })
    elif weight_lb:
        # TODO: Convert pounds to kg and store the result
        lb_to_kg = 2.205
        vitals.append({
            'PATIENT': patient_id,
            'ENCOUNTER': encounter_id,
            'CODE': VITAL_LOINC_CODES['Body Weight'],
            'DESCRIPTION': 'Body Weight',
            'VALUE': round(float(weight_lb.group(1)) / 2.205, 2),
            'UNITS': 'kg'
        })

    # --- SpO2 ---
    spo2_match = re.search(VITAL_PATTERNS['spo2'], note_text, re.IGNORECASE)
    if spo2_match:
        vitals.append({
            'PATIENT': patient_id,
            'ENCOUNTER': encounter_id,
            'CODE': VITAL_LOINC_CODES['Oxygen saturation'],
            'DESCRIPTION': 'Oxygen saturation',
            'VALUE': int(spo2_match.group(1)),
            'UNITS': '%'
        })

    # --- Temperature ---
    temp_match = re.search(VITAL_PATTERNS['temp'], note_text)
    
    if temp_match:
        raw_value = float(temp_match.group(1))
        
        # Check if valid Celsius reading
        if 25.0 <= raw_value <= 45.0:
            final_value = raw_value
            is_valid = True
        # Check if valid Fahrenheit reading
        elif 77.0 <= raw_value <= 115.0:
            # Convert to C
            final_value = (raw_value - 32) * (5/9)
            is_valid = True
        else:
            # Probably invalid
            is_valid = False

        if is_valid:
            vitals.append({
                'PATIENT': patient_id,
                'ENCOUNTER': encounter_id,
                'CODE': VITAL_LOINC_CODES['Body Temperature'],
                'DESCRIPTION': 'Body Temperature',
                'VALUE': round(final_value, 1),
                'UNITS': 'C'
            })

    return vitals


def extract_labs_from_note(note_text, patient_id, encounter_id):
    """
    Extract lab values from clinical note text using LAB_PATTERNS.
    Only returns values within valid physiological ranges.

    Args:
        note_text: The clinical note text to extract from
        patient_id: Patient ID to include in extracted records
        encounter_id: Encounter ID to include in extracted records

    Returns:
        List of dicts, each containing a lab value record with:
        PATIENT, ENCOUNTER, CODE, DESCRIPTION, VALUE, UNITS
    """
    labs = []
    if not isinstance(note_text, str):
        return labs

    for lab_name, pattern in LAB_PATTERNS.items():
        # Skip if pattern is None (not yet implemented)
        if pattern is None:
            continue

        # TODO: Search for the pattern in note_text
        match = re.search(pattern, note_text, re.IGNORECASE)

        if match:
            # TODO: Extract the numeric value from the match
            value = float(match.group(1))

            min_val, max_val = LAB_RANGES[lab_name]

            # Only accept values within valid range
            if value is not None and min_val <= value <= max_val:
                code, description, units = LAB_LOINC_CODES[lab_name]
                # TODO: Create the lab record dict
                labs.append({
                    'PATIENT': patient_id,
                    'ENCOUNTER': encounter_id,
                    'CODE': code,
                    'DESCRIPTION': description,
                    'VALUE': value,
                    'UNITS': units
                })

    return labs


def extract_medications_from_note(note_text, patient_id, encounter_id):
    """
    Extract medications from clinical note text using dictionary matching.

    Args:
        note_text: The clinical note text to extract from
        patient_id: Patient ID to include in extracted records
        encounter_id: Encounter ID to include in extracted records

    Returns:
        List of dicts, each containing a medication record with:
        PATIENT, ENCOUNTER, DESCRIPTION, REASONDESCRIPTION
    """
    medications = []
    if not isinstance(note_text, str):
        return medications

    # TODO: Convert note_text to lowercase for case-insensitive matching
    note_lower = note_text.lower()

    # Track which medications we've found (avoid duplicates)
    found_meds = set()

    for keyword, description in KNOWN_MEDICATIONS.items():
        # TODO: Check if keyword is in note_lower and not already found
        if keyword in note_lower and description not in found_meds: 
            found_meds.add(description)
            # TODO: Create the medication record dict
            medications.append({
                'PATIENT': patient_id,
                'ENCOUNTER': encounter_id,
                'DESCRIPTION': description,
                'REASONDESCRIPTION': None # unsure how to apply reason
            })

    return medications


def extract_all_from_note(note_text, patient_id='', encounter_id=''):
    """
    Extract all structured data from a clinical note.

    This function REUSES the helper functions we built:
    - extract_vitals_from_note()
    - extract_labs_from_note()
    - extract_medications_from_note()

    Args:
        note_text: The clinical note text to extract from
        patient_id: Optional patient ID for the extracted records
        encounter_id: Optional encounter ID for the extracted records

    Returns:
        dict with:
        - vitals: list of vital sign dicts
        - labs: list of lab value dicts
        - medications: list of medication dicts
    """
    if not isinstance(note_text, str):
        return {'vitals': [], 'labs': [], 'medications': []}

    # TODO: Call each extraction function we built above and store results
    result = {
        'vitals': extract_vitals_from_note(note_text, patient_id, encounter_id),
        'labs': extract_labs_from_note(note_text, patient_id, encounter_id),
        'medications': extract_medications_from_note(note_text, patient_id, encounter_id)
    }

    return result
