"""
PatientProfile class for Part 1: Data Extraction Challenge

Students use this class to extract missing information from clinical notes
and combine it with CSV data to create complete patient profiles.
"""

from datetime import datetime
from typing import Dict, List, Set
import pandas as pd


# Diabetes-related terms for identification
DIABETES_TERMS = [
    'diabetes', 'diabetic', 'dm2', 'dm1', 't2dm', 't1dm',
    'type 2 diabetes', 'type 1 diabetes', 'diabetes mellitus',
    'type ii diabetes', 'type i diabetes', 'iddm', 'niddm'
]


def normalize_medical_text(text: str) -> str:
    """Normalize medical text for comparison."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def has_diabetes_condition(condition_data: Dict) -> bool:
    """Check if a condition dictionary indicates diabetes."""
    desc = normalize_medical_text(condition_data.get('description', ''))
    return any(term in desc for term in DIABETES_TERMS)


def determine_diabetes_type(description: str) -> str:
    """Determine diabetes type from description."""
    desc = normalize_medical_text(description)
    if 'type 1' in desc or 't1dm' in desc or 'type i' in desc:
        return 'Type 1'
    elif 'type 2' in desc or 't2dm' in desc or 'type ii' in desc:
        return 'Type 2'
    elif 'gestational' in desc:
        return 'Gestational'
    return 'Unknown'


def validate_hba1c_value(value: float) -> bool:
    """Validate HbA1c value is in reasonable range (4-15%)."""
    return 4.0 <= value <= 15.0


def classify_hba1c_control(value: float) -> str:
    """Classify glycemic control level from HbA1c."""
    if value < 5.7:
        return 'normal'
    elif value < 6.5:
        return 'prediabetes'
    elif value < 7.0:
        return 'well_controlled'
    elif value < 8.0:
        return 'moderate_control'
    elif value < 9.0:
        return 'poor_control'
    else:
        return 'very_poor_control'


class PatientProfile:
    """
    Patient profile class for students to fill from CSV + clinical notes.

    This class represents a patient's comprehensive medical information
    extracted from Synthea CSV data and clinical notes.

    Attributes:
    -----------
    patient_id : str
        Unique patient identifier
    demographics : dict
        Basic demographic info (birthdate, gender, race, etc.)
    conditions : list
        List of medical conditions
    medications : list
        List of medications
    observations : list
        List of clinical observations (lab values, vitals)
    encounters : list
        List of medical encounters
    diabetes_profile : dict
        Diabetes-specific information

    Example:
    --------
    >>> profile = PatientProfile("patient_123", {"birthdate": "1970-01-15", "gender": "M"})
    >>> profile.add_condition({"description": "Type 2 Diabetes Mellitus", "onset": "2020-01-01"})
    >>> profile.diabetes_profile['has_diabetes']
    True
    """

    def __init__(self, patient_id: str, demographics: Dict):
        """
        Initialize patient profile with basic demographics.

        Parameters:
        -----------
        patient_id : str
            Unique patient identifier
        demographics : dict
            Dictionary with basic demographic info
        """
        self.patient_id = patient_id
        self.demographics = demographics

        # Core medical information (students will populate these)
        self.conditions: List[Dict] = []
        self.medications: List[Dict] = []
        self.allergies: List[Dict] = []
        self.procedures: List[Dict] = []
        self.observations: List[Dict] = []
        self.encounters: List[Dict] = []

        # Structured profiles (students may use these)
        self.chronic_conditions: Set[str] = set()
        self.family_history: Dict[str, str] = {}
        self.social_history: Dict[str, str] = {}

        # Diabetes-specific information
        self.diabetes_profile: Dict = {
            'has_diabetes': False,
            'diabetes_type': None,
            'diagnosis_date': None,
            'current_medications': [],
            'latest_hba1c': None,
            'complications': []
        }

    def add_condition(self, condition_data: Dict) -> None:
        """Add a medical condition to the profile."""
        self.conditions.append(condition_data)

        # Auto-detect diabetes
        if has_diabetes_condition(condition_data):
            self.diabetes_profile['has_diabetes'] = True
            self.diabetes_profile['diabetes_type'] = determine_diabetes_type(
                condition_data.get('description', '')
            )
            if not self.diabetes_profile['diagnosis_date']:
                self.diabetes_profile['diagnosis_date'] = condition_data.get('onset', '')

    def add_medication(self, medication_data: Dict) -> None:
        """Add a medication to the profile."""
        self.medications.append(medication_data)

        # Auto-detect diabetes medications
        med_desc = normalize_medical_text(medication_data.get('description', ''))
        diabetes_meds = ['metformin', 'insulin', 'glipizide', 'glyburide',
                         'sitagliptin', 'empagliflozin', 'liraglutide', 'semaglutide']
        if any(med in med_desc for med in diabetes_meds):
            if medication_data not in self.diabetes_profile['current_medications']:
                self.diabetes_profile['current_medications'].append(medication_data)

    def add_observation(self, observation_data: Dict) -> None:
        """Add a clinical observation to the profile."""
        self.observations.append(observation_data)

        # Auto-detect HbA1c values
        obs_desc = normalize_medical_text(observation_data.get('description', ''))
        if 'hba1c' in obs_desc or 'hemoglobin a1c' in obs_desc:
            try:
                value = float(observation_data.get('value', 0))
                if validate_hba1c_value(value):
                    obs_date = observation_data.get('date', '')
                    if (not self.diabetes_profile['latest_hba1c'] or
                            obs_date > self.diabetes_profile['latest_hba1c'].get('date', '')):
                        self.diabetes_profile['latest_hba1c'] = {
                            'value': value,
                            'date': obs_date,
                            'units': observation_data.get('units', '%'),
                            'control_level': classify_hba1c_control(value)
                        }
            except (ValueError, TypeError):
                pass

    def add_encounter_record(self, encounter_data: Dict) -> None:
        """Add a medical encounter to the profile from raw data dict."""
        self.encounters.append(encounter_data)

    def add_procedure(self, procedure_data: Dict) -> None:
        """Add a medical procedure to the profile."""
        self.procedures.append(procedure_data)

    def add_allergy(self, allergy_data: Dict) -> None:
        """Add an allergy to the profile."""
        self.allergies.append(allergy_data)

    def get_current_age(self, reference_date: str = None) -> int:
        """
        Calculate patient's current age.

        Parameters:
        -----------
        reference_date : str, optional
            Date to calculate age from (default: today)

        Returns:
        --------
        int : Patient age in years
        """
        birthdate = self.demographics.get('birthdate', '')
        if not birthdate:
            return 0

        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')

        birthdate_dt = pd.to_datetime(birthdate)
        ref_dt = pd.to_datetime(reference_date)

        age = ref_dt.year - birthdate_dt.year
        if (ref_dt.month, ref_dt.day) < (birthdate_dt.month, birthdate_dt.day):
            age -= 1

        return age

    def has_chronic_condition(self, condition_name: str) -> bool:
        """Check if patient has a specific chronic condition."""
        normalized_name = normalize_medical_text(condition_name)
        for condition in self.conditions:
            condition_desc = normalize_medical_text(condition.get('description', ''))
            if normalized_name in condition_desc:
                return True
        return False

    def get_medications_by_type(self, medication_type: str) -> List[Dict]:
        """Get medications of a specific type (e.g., 'diabetes', 'hypertension')."""
        matching_meds = []
        normalized_type = normalize_medical_text(medication_type)

        for med in self.medications:
            med_desc = normalize_medical_text(med.get('description', ''))
            if normalized_type in med_desc:
                matching_meds.append(med)

        return matching_meds

    def validate_completeness(self) -> Dict[str, bool]:
        """
        Validate that the profile has been filled with essential information.

        Returns:
        --------
        dict : Dictionary indicating which sections are complete
        """
        return {
            'demographics': bool(self.demographics.get('birthdate') and self.demographics.get('gender')),
            'conditions': len(self.conditions) > 0,
            'medications': len(self.medications) > 0,
            'encounters': len(self.encounters) > 0,
            'observations': len(self.observations) > 0
        }

    def to_dict(self) -> Dict:
        """Convert profile to dictionary for serialization."""
        return {
            'patient_id': self.patient_id,
            'demographics': self.demographics,
            'conditions': self.conditions,
            'medications': self.medications,
            'allergies': self.allergies,
            'procedures': self.procedures,
            'observations': self.observations,
            'encounters': self.encounters,
            'chronic_conditions': list(self.chronic_conditions),
            'family_history': self.family_history,
            'social_history': self.social_history,
            'diabetes_profile': self.diabetes_profile
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PatientProfile':
        """Create profile from dictionary."""
        profile = cls(data['patient_id'], data['demographics'])
        profile.conditions = data.get('conditions', [])
        profile.medications = data.get('medications', [])
        profile.allergies = data.get('allergies', [])
        profile.procedures = data.get('procedures', [])
        profile.observations = data.get('observations', [])
        profile.encounters = data.get('encounters', [])
        profile.chronic_conditions = set(data.get('chronic_conditions', []))
        profile.family_history = data.get('family_history', {})
        profile.social_history = data.get('social_history', {})
        profile.diabetes_profile = data.get('diabetes_profile', {})
        return profile

    def __repr__(self) -> str:
        return f"PatientProfile(id={self.patient_id}, conditions={len(self.conditions)}, encounters={len(self.encounters)})"
