"""
ExtendedPatientProfile class for Part 2: Agentic Risk Classification

This class extends PatientProfile with time-series timeline attributes
for tracking patient events over time. In Week 2, you will implement
feature engineering methods using these timelines.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from .patient_profile import PatientProfile


def safe_parse_date(date_str) -> datetime:
    """Safely parse a date string to datetime."""
    if isinstance(date_str, datetime):
        return date_str
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except:
        return None


class ExtendedPatientProfile(PatientProfile):
    """
    Extended patient profile with time-series capabilities for diabetes risk classification.

    This class extends the basic PatientProfile with:
    - Time-series event tracking (timelines)
    - Methods to add events to timelines
    - Temporal safety utilities

    Attributes:
    -----------
    hba1c_timeline : list
        Time-series of HbA1c readings
    medication_timeline : list
        Time-series of medication changes
    emergency_visit_timeline : list
        Time-series of ED visits
    hospitalization_timeline : list
        Time-series of hospitalizations
    encounter_timeline : list
        Time-series of all encounters
    risk_events : list
        High-risk events for labeling

    Example:
    --------
    >>> profile = ExtendedPatientProfile("patient_123", demographics)
    >>> profile.add_hba1c_reading("2024-01-15", 7.2)
    >>> profile.add_encounter("2024-02-01", "wellness", "Annual checkup")
    """

    def __init__(self, patient_id: str, demographics: Dict):
        super().__init__(patient_id, demographics)

        # Time-series event timelines (students populate these)
        self.hba1c_timeline: List[Dict] = []
        self.medication_timeline: List[Dict] = []
        self.emergency_visit_timeline: List[Dict] = []
        self.hospitalization_timeline: List[Dict] = []
        self.care_gap_timeline: List[Dict] = []
        self.complication_timeline: List[Dict] = []
        self.encounter_timeline: List[Dict] = []

        # Additional observation timelines
        self.bp_timeline: List[Dict] = []  # Blood pressure readings
        self.egfr_timeline: List[Dict] = []  # eGFR readings
        self.bmi_timeline: List[Dict] = []  # BMI readings
        self.missed_appointments: List[Dict] = []  # Missed/cancelled appointments

        # Risk assessment attributes
        self.risk_events: List[Dict] = []
        self.current_risk_status: str = "unknown"

    def add_hba1c_reading(self, date: str, value: float, units: str = "%") -> None:
        """Add HbA1c reading to timeline."""
        self.hba1c_timeline.append({
            "date": date,
            "value": value,
            "units": units
        })
        self.hba1c_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_medication_event(self, date: str, action: str, medication: str, dosage: str = "") -> None:
        """
        Add medication change event to timeline.

        Parameters:
        -----------
        date : str
            Date of medication change
        action : str
            "started", "stopped", "increased", "decreased", "continued"
        medication : str
            Medication name
        dosage : str, optional
            Dosage information
        """
        self.medication_timeline.append({
            "date": date,
            "action": action,
            "medication": medication,
            "dosage": dosage
        })
        self.medication_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_emergency_visit(self, date: str, reason: str, diabetes_related: bool = False) -> None:
        """Add emergency department visit to timeline."""
        self.emergency_visit_timeline.append({
            "date": date,
            "reason": reason,
            "diabetes_related": diabetes_related
        })
        self.emergency_visit_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_hospitalization(self, date: str, reason: str, diabetes_related: bool = False,
                            length_of_stay: int = 1) -> None:
        """Add hospitalization to timeline."""
        self.hospitalization_timeline.append({
            "date": date,
            "reason": reason,
            "diabetes_related": diabetes_related,
            "length_of_stay": length_of_stay
        })
        self.hospitalization_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_encounter(self, date: str, encounter_type: str, reason: str = "") -> None:
        """Add general encounter to timeline and base encounter list."""
        encounter_dict = {
            "date": date,
            "encounter_type": encounter_type,
            "reason": reason
        }
        self.encounter_timeline.append(encounter_dict)
        self.encounter_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)
        # Also populate parent's encounters list for summary() compatibility
        self.encounters.append(encounter_dict)

    def add_encounter_event(self, date: str, encounter_type: str, reason: str = "") -> None:
        """Alias for add_encounter."""
        self.add_encounter(date, encounter_type, reason)

    def add_complication(self, date: str, complication: str, severity: str = "unknown") -> None:
        """Add diabetes complication to timeline."""
        self.complication_timeline.append({
            "date": date,
            "complication": complication,
            "severity": severity
        })
        self.complication_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_care_gap(self, date: str, gap_type: str, days_overdue: int) -> None:
        """Add care gap event to timeline."""
        self.care_gap_timeline.append({
            "date": date,
            "gap_type": gap_type,
            "days_overdue": days_overdue
        })
        self.care_gap_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_bp_reading(self, date: str, systolic: float, diastolic: float = None) -> None:
        """Add blood pressure reading to timeline."""
        self.bp_timeline.append({
            "date": date,
            "systolic": systolic,
            "diastolic": diastolic
        })
        self.bp_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_egfr_reading(self, date: str, value: float) -> None:
        """Add eGFR reading to timeline."""
        self.egfr_timeline.append({
            "date": date,
            "value": value
        })
        self.egfr_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_bmi_reading(self, date: str, value: float) -> None:
        """Add BMI reading to timeline."""
        self.bmi_timeline.append({
            "date": date,
            "value": value
        })
        self.bmi_timeline.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_missed_appointment(self, date: str, reason: str = "") -> None:
        """Add missed/cancelled appointment to timeline."""
        self.missed_appointments.append({
            "date": date,
            "reason": reason
        })
        self.missed_appointments.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def add_risk_event(self, date: str, event_type: str, severity: str = "high") -> None:
        """
        Add a high-risk event for retroactive risk labeling.

        Parameters:
        -----------
        date : str
            Date of high-risk event
        event_type : str
            Type of event (e.g., "emergency_visit", "hospitalization", "ketoacidosis")
        severity : str
            Severity level ("low", "moderate", "high")
        """
        self.risk_events.append({
            "date": date,
            "event_type": event_type,
            "severity": severity
        })
        self.risk_events.sort(key=lambda x: safe_parse_date(x['date']) or datetime.min)

    def timeline_summary(self) -> Dict[str, int]:
        """
        Get summary of timeline data counts.

        Returns:
        --------
        dict : Dictionary with counts for each timeline
        """
        return {
            'hba1c_readings': len(self.hba1c_timeline),
            'medication_events': len(self.medication_timeline),
            'emergency_visits': len(self.emergency_visit_timeline),
            'hospitalizations': len(self.hospitalization_timeline),
            'encounters': len(self.encounter_timeline),
            'complications': len(self.complication_timeline),
            'care_gaps': len(self.care_gap_timeline),
            'risk_events': len(self.risk_events)
        }

    def validate_timelines(self) -> Dict[str, bool]:
        """
        Validate that time-series data has been properly populated.

        Returns:
        --------
        dict : Dictionary indicating which timelines have data
        """
        return {
            'hba1c_timeline': len(self.hba1c_timeline) > 0,
            'medication_timeline': len(self.medication_timeline) > 0,
            'emergency_visit_timeline': len(self.emergency_visit_timeline) > 0,
            'hospitalization_timeline': len(self.hospitalization_timeline) > 0,
            'care_gap_timeline': len(self.care_gap_timeline) > 0,
            'complication_timeline': len(self.complication_timeline) > 0,
            'encounter_timeline': len(self.encounter_timeline) > 0,
            'risk_events': len(self.risk_events) > 0
        }

    def to_dict(self) -> Dict:
        """Convert extended profile to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'hba1c_timeline': self.hba1c_timeline,
            'medication_timeline': self.medication_timeline,
            'emergency_visit_timeline': self.emergency_visit_timeline,
            'hospitalization_timeline': self.hospitalization_timeline,
            'care_gap_timeline': self.care_gap_timeline,
            'complication_timeline': self.complication_timeline,
            'encounter_timeline': self.encounter_timeline,
            'risk_events': self.risk_events,
            'current_risk_status': self.current_risk_status
        })
        return base_dict

    @classmethod
    def from_patient_profile(cls, profile: PatientProfile) -> 'ExtendedPatientProfile':
        """Create ExtendedPatientProfile from basic PatientProfile."""
        extended = cls(profile.patient_id, profile.demographics)

        # Copy all basic profile data
        extended.conditions = profile.conditions
        extended.medications = profile.medications
        extended.allergies = profile.allergies
        extended.procedures = profile.procedures
        extended.observations = profile.observations
        extended.encounters = profile.encounters
        extended.chronic_conditions = profile.chronic_conditions
        extended.family_history = profile.family_history
        extended.social_history = profile.social_history
        extended.diabetes_profile = profile.diabetes_profile

        return extended

    def summary(self) -> None:
        """Print a summary of the extended patient profile including timeline data."""
        super().summary()
        print(f"--- Timelines ---")
        print(f"HbA1c readings: {len(self.hba1c_timeline)}, BP readings: {len(self.bp_timeline)}")
        print(f"Medication events: {len(self.medication_timeline)}, eGFR readings: {len(self.egfr_timeline)}")
        print(f"Emergency visits: {len(self.emergency_visit_timeline)}, Hospitalizations: {len(self.hospitalization_timeline)}")
        print(f"Complications: {len(self.complication_timeline)}, Care gaps: {len(self.care_gap_timeline)}")
        print(f"BMI readings: {len(self.bmi_timeline)}, Risk events: {len(self.risk_events)}")

    def __repr__(self) -> str:
        return (f"ExtendedPatientProfile(id={self.patient_id}, "
                f"encounters={len(self.encounter_timeline)}, "
                f"hba1c_readings={len(self.hba1c_timeline)})")
