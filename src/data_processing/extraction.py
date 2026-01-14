"""Clinical note extraction functions for vitals, labs, and medications."""

import re

# =============================================================================
# VITAL SIGNS PATTERNS AND CODES
# =============================================================================

VITAL_PATTERNS = {
    'bp': r'(?:BP|Blood Pressure)[:\s]*(\d{2,3})/(\d{2,3})',
    'weight_kg': r'(?:Weight|Wt)[:\s]*(\d+\.?\d*)\s*(?:kg|KG)',
    'weight_lb': r'(?:Weight|Wt)[:\s]*(\d+\.?\d*)\s*(?:lb|lbs|LB)',
    'height_cm': r'(?:Height|Ht)[:\s]*(\d+\.?\d*)\s*(?:cm|CM)',
    'spo2': r'(?:SpO2|O2 Sat|Oxygen)[:\s]*(\d{2,3})\s*%',
    'temp': None  # TODO: Add temperature pattern
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
    'egfr': None  # TODO: Add eGFR pattern
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
    'metformin': 'Metformin',
    'glipizide': 'Glipizide',
    'glyburide': 'Glyburide',
    'glimepiride': 'Glimepiride',
    'insulin': 'Insulin',
    'lantus': 'Insulin Glargine (Lantus)',
    'humalog': 'Insulin Lispro (Humalog)',
    'novolog': 'Insulin Aspart (Novolog)',
    'jardiance': 'Empagliflozin (Jardiance)',
    'farxiga': 'Dapagliflozin (Farxiga)',
    'ozempic': 'Semaglutide (Ozempic)',
    'trulicity': 'Dulaglutide (Trulicity)',
    'januvia': 'Sitagliptin (Januvia)',
    'lisinopril': 'Lisinopril',
    'atorvastatin': 'Atorvastatin',
    'amlodipine': 'Amlodipine',
    'omeprazole': 'Omeprazole',
    'aspirin': 'Aspirin'
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

    # TODO: Implement extraction for BP, Weight, SpO2, Temperature
    # Use VITAL_PATTERNS and VITAL_LOINC_CODES

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

    # TODO: Implement extraction for HbA1c, Glucose, Creatinine, eGFR
    # Use LAB_PATTERNS, LAB_LOINC_CODES, and LAB_RANGES for validation

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

    # TODO: Implement dictionary-based medication matching
    # Use KNOWN_MEDICATIONS dict

    return medications


def extract_all_from_note(note_text, patient_id='', encounter_id=''):
    """
    Extract all structured data from a clinical note.

    This function combines the individual extraction functions:
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

    # TODO: Call extraction functions and return combined results
    return {
        'vitals': None,
        'labs': None,
        'medications': None
    }
