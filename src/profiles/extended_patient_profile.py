"""
ExtendedPatientProfile class for Part 2: Agentic Risk Classification

Students use this class to create time-series datasets for ML training
and implement classifier features for diabetes risk prediction.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from .patient_profile import PatientProfile, DIABETES_TERMS


# High-Risk Event Definitions (ADA 2025 Standards)
ED_VISIT_CODE = '50849002'
INPATIENT_CODES = ['32485007', '183452005', '305351004']
ACUTE_COMPLICATION_TERMS = [
    'ketoacidosis', 'dka', 'hypoglycemia', 'hypoglycemic',
    'myocardial infarction', 'heart attack', 'stroke', 'cerebrovascular',
    'amputation', 'gangrene', 'emergency', 'inpatient', 'hospitalization'
]
HBA1C_POOR_CONTROL = 10.0


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
    - Time-series event tracking
    - Feature engineering methods for ML
    - Risk assessment utilities

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
    >>> features = profile.get_daily_features("2024-06-01")
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

        # Additional observation timelines for Session 1 features
        self.bp_timeline: List[Dict] = []  # Blood pressure readings
        self.egfr_timeline: List[Dict] = []  # eGFR readings
        self.bmi_timeline: List[Dict] = []  # BMI readings

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

    # ──────────────────────────────────────────────────────────────
    # PROVIDED Helper Methods (use these in your implementations)
    # ──────────────────────────────────────────────────────────────

    def _days_since_last_event(self, target_date: datetime, timeline: List[Dict]) -> Optional[int]:
        """Calculate days since last event in timeline before target_date."""
        events_before = [
            e for e in timeline
            if safe_parse_date(e['date']) and safe_parse_date(e['date']) <= target_date
        ]

        if not events_before:
            return None

        last_event = max(events_before, key=lambda x: safe_parse_date(x['date']))
        last_date = safe_parse_date(last_event['date'])
        return (target_date - last_date).days

    def _count_events_in_window(self, target_date: datetime, timeline: List[Dict],
                                 window_days: int) -> int:
        """Count events in timeline within window_days before target_date."""
        window_start = target_date - timedelta(days=window_days)

        count = 0
        for event in timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and window_start <= event_date <= target_date:
                count += 1

        return count

    def _get_most_recent_hba1c(self, target_date: datetime) -> Optional[float]:
        """Get most recent HbA1c value before target_date."""
        readings_before = [
            r for r in self.hba1c_timeline
            if safe_parse_date(r['date']) and safe_parse_date(r['date']) <= target_date
        ]

        if not readings_before:
            return None

        last_reading = max(readings_before, key=lambda x: safe_parse_date(x['date']))
        return last_reading.get('value')

    # ──────────────────────────────────────────────────────────────
    # SESSION 1: Feature methods (6 provided implementations)
    # ──────────────────────────────────────────────────────────────

    def _days_since_last_hba1c(self, target_date: datetime) -> Optional[int]:
        """Days since most recent HbA1c reading before target_date."""
        return self._days_since_last_event(target_date, self.hba1c_timeline)

    def _current_hba1c_level(self, target_date: datetime) -> Optional[float]:
        """Most recent HbA1c value (%) before target_date."""
        return self._get_most_recent_hba1c(target_date)

    def _encounters_last_90d(self, target_date: datetime) -> int:
        """Number of encounters in the 90 days before target_date."""
        return self._count_events_in_window(target_date, self.encounter_timeline, 90)

    def _age_at_date(self, date: str) -> Optional[int]:
        """Patient age at the given date."""
        return self.get_current_age(date)

    def _current_systolic_bp(self, target_date: datetime) -> Optional[float]:
        """Most recent systolic blood pressure (mmHg) before target_date."""
        return self._get_most_recent_bp(target_date)

    def _current_egfr(self, target_date: datetime) -> Optional[float]:
        """Most recent eGFR (mL/min/1.73m2) before target_date."""
        return self._get_most_recent_egfr(target_date)

    # ──────────────────────────────────────────────────────────────
    # HOMEWORK TODO: Implement these 8 feature methods
    # ──────────────────────────────────────────────────────────────

    def _days_since_last_encounter(self, target_date: datetime) -> Optional[int]:
        """
        Calculate days since last encounter before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Use self.encounter_timeline and filter to events before target_date.
        You can use the _days_since_last_event helper method, or implement directly.

        Returns:
        --------
        int : Days since last encounter, or None if no encounters found
        """
        # TODO: Implement this method
        return None

    def _days_since_medication_change(self, target_date: datetime) -> Optional[int]:
        """
        Calculate days since last medication change before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Use self.medication_timeline and filter to events before target_date.
        You can use the _days_since_last_event helper method, or implement directly.

        Returns:
        --------
        int : Days since last medication change, or None if no changes found
        """
        # TODO: Implement this method
        return None

    def _emergency_visits_last_180d(self, target_date: datetime) -> int:
        """
        Count emergency visits in the 180 days before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Use self.emergency_visit_timeline and count events within the window.
        You can use the _count_events_in_window helper method, or implement directly.

        Returns:
        --------
        int : Number of emergency visits in the last 180 days
        """
        # TODO: Implement this method
        return 0

    def _medication_changes_last_90d(self, target_date: datetime) -> int:
        """
        Count medication changes in the 90 days before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Use self.medication_timeline and count events within the window.
        You can use the _count_events_in_window helper method, or implement directly.

        Returns:
        --------
        int : Number of medication changes in the last 90 days
        """
        # TODO: Implement this method
        return 0

    def _calculate_hba1c_trend(self, target_date: datetime, window_days: int = 180) -> float:
        """
        Calculate HbA1c trend (slope) over window_days before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Get readings in the window, compare first and last values.
        A change > 0.5 is worsening, < -0.5 is improving, else stable.

        Returns:
        --------
        float : Trend direction (-1=improving, 0=stable, 1=worsening)
        """
        # TODO: Implement this method
        return 0

    def _get_bmi_category(self, target_date: datetime) -> Optional[int]:
        """
        Get BMI category from most recent BMI reading before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Use self.bmi_timeline to find the most recent BMI reading.
        Then classify it: <18.5=1 (underweight), 18.5-24.9=2 (normal),
        25-29.9=3 (overweight), >=30=4 (obese).

        Returns:
        --------
        int : BMI category (1-4), or None if no BMI data found
        """
        # TODO: Implement this method
        return None

    def _get_active_medication_count(self, target_date: datetime) -> int:
        """
        Count medications active at target_date.

        HOMEWORK TODO: Implement this method.

        A medication is active if: started before target_date AND
        (not stopped OR stopped after target_date).

        Hint: Use self.medication_timeline. Each entry has 'action' ('started'
        or 'stopped'), 'medication' (name), and 'date'. Group by medication name,
        check if its start is before target_date and its stop (if any) is after.

        Returns:
        --------
        int : Number of active medications
        """
        # TODO: Implement this method
        return 0

    def _get_longest_care_gap(self, target_date: datetime) -> Optional[int]:
        """
        Get longest gap (days) between consecutive HbA1c tests before target_date.

        HOMEWORK TODO: Implement this method.

        Hint: Use self.hba1c_timeline. Filter to readings before target_date,
        sort by date, compute gaps between consecutive readings, return the max.
        Return None if fewer than 2 readings.

        Returns:
        --------
        int : Longest gap in days, or None if < 2 HbA1c readings
        """
        # TODO: Implement this method
        return None

    # ──────────────────────────────────────────────────────────────
    # Feature Extraction Methods
    # ──────────────────────────────────────────────────────────────

    def get_daily_features(self, date: str) -> Dict:
        """
        Generate feature vector for classifier training.

        Parameters:
        -----------
        date : str
            Date in format "YYYY-MM-DD" (prediction point)

        Returns:
        --------
        dict : Dictionary with features for ML model

        Features Include:
        - Temporal counters (days since events)
        - Rolling window statistics (event counts)
        - Current state indicators (latest values)
        """
        if not self.diabetes_profile.get('has_diabetes'):
            return {}

        target_date = safe_parse_date(date)
        if not target_date:
            return {}

        features = {}

        # Temporal counters (Session 1 features)
        features['age_at_date'] = self._age_at_date(date)
        features['days_since_last_hba1c'] = self._days_since_last_hba1c(target_date)
        # HOMEWORK TODO: These use your implemented methods
        features['days_since_last_encounter'] = self._days_since_last_encounter(target_date)
        features['days_since_medication_change'] = self._days_since_medication_change(target_date)
        # PROVIDED: These use helper methods directly
        features['days_since_emergency_visit'] = self._days_since_last_event(
            target_date, self.emergency_visit_timeline
        )
        features['days_since_hospitalization'] = self._days_since_last_event(
            target_date, self.hospitalization_timeline
        )

        # Rolling window statistics (Session 1 feature)
        features['encounters_last_90d'] = self._encounters_last_90d(target_date)
        # HOMEWORK TODO: These use your implemented methods
        features['emergency_visits_last_180d'] = self._emergency_visits_last_180d(target_date)
        # PROVIDED: This uses helper method directly
        features['hospitalizations_last_365d'] = self._count_events_in_window(
            target_date, self.hospitalization_timeline, 365
        )
        # HOMEWORK TODO: This uses your implemented method
        features['medication_changes_last_90d'] = self._medication_changes_last_90d(target_date)

        # Current state indicators (Session 1 feature)
        features['current_hba1c_level'] = self._current_hba1c_level(target_date)
        features['hba1c_trend_last_180d'] = self._calculate_hba1c_trend(target_date, 180)
        features['diabetes_complication_count'] = self._get_complication_count(target_date)

        # Care gap features
        care_gaps_before = [
            g for g in self.care_gap_timeline
            if safe_parse_date(g['date']) and safe_parse_date(g['date']) <= target_date
        ]
        features['care_gaps_count'] = len(care_gaps_before)
        features['longest_care_gap_days'] = self._get_longest_care_gap(target_date)

        # Blood pressure features (Session 1 feature)
        features['current_systolic_bp'] = self._current_systolic_bp(target_date)
        features['current_diastolic_bp'] = self._get_most_recent_diastolic_bp(target_date)
        features['bp_trend_last_180d'] = self._calculate_bp_trend(target_date, 180)

        # Kidney function features (Session 1 feature)
        features['current_egfr'] = self._current_egfr(target_date)
        features['egfr_trend_last_365d'] = self._calculate_egfr_trend(target_date, 365)

        # Medication features
        features['active_medication_count'] = self._get_active_medication_count(target_date)

        # BMI category feature
        features['bmi_category'] = self._get_bmi_category(target_date)

        return features

    def _get_most_recent_bp(self, target_date: datetime) -> Optional[float]:
        """Get most recent systolic BP value before target_date."""
        readings_before = [
            r for r in self.bp_timeline
            if safe_parse_date(r['date']) and safe_parse_date(r['date']) <= target_date
        ]
        if not readings_before:
            return None
        last_reading = max(readings_before, key=lambda x: safe_parse_date(x['date']))
        return last_reading.get('systolic')

    def _get_most_recent_egfr(self, target_date: datetime) -> Optional[float]:
        """Get most recent eGFR value before target_date."""
        readings_before = [
            r for r in self.egfr_timeline
            if safe_parse_date(r['date']) and safe_parse_date(r['date']) <= target_date
        ]
        if not readings_before:
            return None
        last_reading = max(readings_before, key=lambda x: safe_parse_date(x['date']))
        return last_reading.get('value')

    def _get_most_recent_diastolic_bp(self, target_date: datetime) -> Optional[float]:
        """Get most recent diastolic BP value before target_date."""
        readings_before = [
            r for r in self.bp_timeline
            if safe_parse_date(r['date']) and safe_parse_date(r['date']) <= target_date
        ]
        if not readings_before:
            return None
        last_reading = max(readings_before, key=lambda x: safe_parse_date(x['date']))
        return last_reading.get('diastolic')

    def _calculate_bp_trend(self, target_date: datetime, days: int) -> int:
        """
        Calculate blood pressure trend over specified period.
        Returns: -1 (improving/decreasing), 0 (stable), 1 (worsening/increasing)
        """
        window_start = target_date - timedelta(days=days)
        readings_in_window = [
            r for r in self.bp_timeline
            if safe_parse_date(r['date']) and window_start <= safe_parse_date(r['date']) <= target_date
        ]
        if len(readings_in_window) < 2:
            return 0

        readings_in_window.sort(key=lambda x: safe_parse_date(x['date']))
        first_bp = readings_in_window[0].get('systolic')
        last_bp = readings_in_window[-1].get('systolic')

        if first_bp is not None and last_bp is not None:
            diff = last_bp - first_bp
            if diff >= 10:
                return 1  # Worsening (BP increasing)
            elif diff <= -10:
                return -1  # Improving (BP decreasing)
        return 0  # Stable

    def _calculate_egfr_trend(self, target_date: datetime, days: int) -> int:
        """
        Calculate eGFR trend over specified period.
        Returns: -1 (worsening/decreasing), 0 (stable), 1 (improving/increasing)
        """
        window_start = target_date - timedelta(days=days)
        readings_in_window = [
            r for r in self.egfr_timeline
            if safe_parse_date(r['date']) and window_start <= safe_parse_date(r['date']) <= target_date
        ]
        if len(readings_in_window) < 2:
            return 0

        readings_in_window.sort(key=lambda x: safe_parse_date(x['date']))
        first_egfr = readings_in_window[0].get('value')
        last_egfr = readings_in_window[-1].get('value')

        if first_egfr is not None and last_egfr is not None:
            diff = last_egfr - first_egfr
            if diff <= -5:
                return -1  # Worsening (eGFR decreasing = kidney function declining)
            elif diff >= 5:
                return 1  # Improving (eGFR increasing)
        return 0  # Stable

    def get_session_1_features(self, date: str) -> Dict:
        """
        Get the 6 features defined in Session 1 (TDD-verified).

        Features:
        - days_since_last_hba1c: Days since most recent HbA1c reading
        - current_hba1c_level: Most recent HbA1c value (%)
        - encounters_last_90d: Number of encounters in last 90 days
        - age_at_date: Patient age at this date
        - current_systolic_bp: Most recent systolic BP (mmHg)
        - current_egfr: Most recent eGFR (mL/min/1.73m2)
        """
        target_date = safe_parse_date(date)
        if not target_date:
            return {}

        return {
            'days_since_last_hba1c': self._days_since_last_hba1c(target_date),
            'current_hba1c_level': self._current_hba1c_level(target_date),
            'encounters_last_90d': self._encounters_last_90d(target_date),
            'age_at_date': self._age_at_date(date),
            'current_systolic_bp': self._current_systolic_bp(target_date),
            'current_egfr': self._current_egfr(target_date)
        }

    def get_homework_features(self, date: str) -> Dict:
        """
        Get only the 8 homework features (for testing your TODO implementations).

        Features:
        - days_since_last_encounter: Days since most recent encounter
        - days_since_medication_change: Days since last medication change
        - emergency_visits_last_180d: Number of ED visits in last 180 days
        - medication_changes_last_90d: Number of medication changes in last 90 days
        - hba1c_trend_last_180d: HbA1c trend (-1=improving, 0=stable, 1=worsening)
        - active_medication_count: Number of active medications
        - longest_care_gap_days: Longest care gap in days
        - bmi_category: BMI category (1-4)
        """
        target_date = safe_parse_date(date)
        if not target_date:
            return {}

        return {
            'days_since_last_encounter': self._days_since_last_encounter(target_date),
            'days_since_medication_change': self._days_since_medication_change(target_date),
            'emergency_visits_last_180d': self._emergency_visits_last_180d(target_date),
            'medication_changes_last_90d': self._medication_changes_last_90d(target_date),
            'hba1c_trend_last_180d': self._calculate_hba1c_trend(target_date, 180),
            'active_medication_count': self._get_active_medication_count(target_date),
            'longest_care_gap_days': self._get_longest_care_gap(target_date),
            'bmi_category': self._get_bmi_category(target_date)
        }

    def check_high_risk_event_ada(self, start_date: str, end_date: str) -> int:
        """
        Check for high-risk events using ADA 2025 definitions.

        Returns 1 if ANY of the following occur in (start_date, end_date]:
        1. ED visit (diabetes-related)
        2. Hospitalization
        3. Acute complication (DKA, hypoglycemia, cardiovascular)
        4. Poor glycemic control (HbA1c ≥10%)
        """
        start_dt = safe_parse_date(start_date)
        end_dt = safe_parse_date(end_date)
        if not start_dt or not end_dt:
            return 0

        # Check emergency visits
        for event in self.emergency_visit_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and start_dt < event_date <= end_dt:
                return 1

        # Check hospitalizations
        for event in self.hospitalization_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and start_dt < event_date <= end_dt:
                return 1

        # Check for acute complications in encounters
        for event in self.encounter_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and start_dt < event_date <= end_dt:
                reason = str(event.get('reason', '')).lower()
                enc_type = str(event.get('encounter_type', '')).lower()
                if any(term in reason for term in ACUTE_COMPLICATION_TERMS):
                    return 1
                if any(term in enc_type for term in ACUTE_COMPLICATION_TERMS):
                    return 1

        # Check for HbA1c >= 10%
        for reading in self.hba1c_timeline:
            reading_date = safe_parse_date(reading['date'])
            if reading_date and start_dt < reading_date <= end_dt:
                if reading.get('value', 0) >= HBA1C_POOR_CONTROL:
                    return 1

        return 0

    def generate_all_instances(self, interval_days: int = 7) -> List[Dict]:
        """
        Generate training instances with ALL 22 features + survival columns.
        Uses helper functions for feature computation.
        FAST: Dates are pre-parsed during loading.

        Includes survival analysis columns:
        - days_until_event: Days from this date to next high-risk event
        - event_observed: 1 if event occurs, 0 if censored
        """
        if not self.encounter_timeline:
            return []

        # Get encounter dates (already datetime objects from loading)
        enc_dates = [e['date'] for e in self.encounter_timeline if e['date'] is not None]
        if len(enc_dates) < 2:
            return []

        min_date, max_date = min(enc_dates), max(enc_dates)
        if (max_date - min_date).days < 60:
            return []

        instances = []
        cur = min_date + timedelta(days=30)
        end = max_date - timedelta(days=30)

        while cur <= end:
            date_str = cur.strftime('%Y-%m-%d')

            # Get survival data (days until event, event observed)
            survival_data = self._get_survival_data(cur, max_date)

            # Use dedicated feature methods for all features
            instances.append({
                'patient_id': self.patient_id,
                'date': date_str,
                # Classification target
                'will_have_high_risk_event_next_30d': 1 if self.is_high_risk_on_date(date_str, 30) else 0,
                # Survival analysis targets
                'survival_days_until_event': survival_data['survival_days_until_event'],
                'survival_event_observed': survival_data['survival_event_observed'],
                # Session 1 features (6):
                'age_at_date': self._age_at_date(date_str),
                'days_since_last_hba1c': self._days_since_last_hba1c(cur),
                'current_hba1c_level': self._current_hba1c_level(cur),
                'encounters_last_90d': self._encounters_last_90d(cur),
                'current_systolic_bp': self._current_systolic_bp(cur),
                'current_egfr': self._current_egfr(cur),
                # Homework TODO features (8):
                'days_since_last_encounter': self._days_since_last_encounter(cur),
                'days_since_medication_change': self._days_since_medication_change(cur),
                'emergency_visits_last_180d': self._emergency_visits_last_180d(cur),
                'medication_changes_last_90d': self._medication_changes_last_90d(cur),
                'hba1c_trend_last_180d': self._calculate_hba1c_trend(cur, 180),
                'bmi_category': self._get_bmi_category(cur),
                'active_medication_count': self._get_active_medication_count(cur),
                'longest_care_gap_days': self._get_longest_care_gap(cur),
                # Other provided features:
                'days_since_emergency_visit': self._days_since_last_event(cur, self.emergency_visit_timeline),
                'days_since_hospitalization': self._days_since_last_event(cur, self.hospitalization_timeline),
                'hospitalizations_last_365d': self._count_events_in_window(cur, self.hospitalization_timeline, 365),
                'current_diastolic_bp': self._get_most_recent_diastolic_bp(cur),
                'bp_trend_last_180d': self._calculate_bp_trend(cur, 180),
                'egfr_trend_last_365d': self._calculate_egfr_trend(cur, 365),
                'diabetes_complication_count': self._get_complication_count(cur),
                'care_gaps_count': self._get_care_gaps_count(cur),
            })

            cur += timedelta(days=interval_days)

        return instances

    def _get_complication_count(self, target_date: datetime) -> int:
        """Get count of diabetes complications diagnosed before target_date."""
        return len([
            c for c in self.complication_timeline
            if safe_parse_date(c['date']) and safe_parse_date(c['date']) <= target_date
        ])

    def _get_care_gaps_count(self, target_date: datetime) -> int:
        """Get count of care gaps before target_date."""
        return len([
            g for g in self.care_gap_timeline
            if safe_parse_date(g['date']) and safe_parse_date(g['date']) <= target_date
        ])

    def _get_next_high_risk_event_date(self, target_date: datetime) -> Optional[datetime]:
        """
        Find the next high-risk event date AFTER target_date.

        High-risk events (ADA 2025):
        - Emergency visits
        - Hospitalizations
        - HbA1c >= 10%

        Returns:
            datetime of next event, or None if no future event
        """
        future_event_dates = []

        # Check emergency visits
        for event in self.emergency_visit_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and event_date > target_date:
                future_event_dates.append(event_date)

        # Check hospitalizations
        for event in self.hospitalization_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and event_date > target_date:
                future_event_dates.append(event_date)

        # Check HbA1c >= 10%
        for reading in self.hba1c_timeline:
            reading_date = safe_parse_date(reading['date'])
            if reading_date and reading_date > target_date:
                if reading.get('value', 0) >= HBA1C_POOR_CONTROL:
                    future_event_dates.append(reading_date)

        if future_event_dates:
            return min(future_event_dates)
        return None

    def _get_survival_data(self, target_date: datetime, max_date: datetime) -> Dict:
        """
        Get survival analysis columns for a given date.

        Returns:
            dict with:
            - days_until_event: Days from target_date to next event (or to max_date if censored)
            - event_observed: 1 if event occurs after target_date, 0 if censored
        """
        next_event_date = self._get_next_high_risk_event_date(target_date)

        if next_event_date and next_event_date <= max_date:
            # Event observed
            days_until = (next_event_date - target_date).days
            return {
                'survival_days_until_event': max(days_until, 1),  # Minimum 1 day
                'survival_event_observed': 1
            }
        else:
            # Censored - no event observed before max_date
            days_until = (max_date - target_date).days
            return {
                'survival_days_until_event': max(days_until, 1),
                'survival_event_observed': 0
            }

    def generate_all_instances_session_1(self, interval_days: int = 1) -> List[Dict]:
        """
        Generate training instances with Session 1's 6 TDD-verified features.
        Uses helper functions for feature computation.
        FAST: Dates are pre-parsed during loading.

        Features (6):
        - days_since_last_hba1c
        - current_hba1c_level
        - encounters_last_90d
        - age_at_date
        - current_systolic_bp
        - current_egfr
        """
        if not self.encounter_timeline:
            return []

        # Get encounter dates (already datetime objects from loading)
        enc_dates = [e['date'] for e in self.encounter_timeline if e['date'] is not None]
        if len(enc_dates) < 2:
            return []

        min_date, max_date = min(enc_dates), max(enc_dates)
        if (max_date - min_date).days < 60:
            return []

        instances = []
        cur = min_date + timedelta(days=30)
        end = max_date - timedelta(days=30)

        while cur <= end:
            date_str = cur.strftime('%Y-%m-%d')

            # Use dedicated Session 1 feature methods
            instances.append({
                'patient_id': self.patient_id,
                'date': date_str,
                'will_have_high_risk_event_next_30d': 1 if self.is_high_risk_on_date(date_str, 30) else 0,
                # Session 1 features (6 total)
                'days_since_last_hba1c': self._days_since_last_hba1c(cur),
                'current_hba1c_level': self._current_hba1c_level(cur),
                'encounters_last_90d': self._encounters_last_90d(cur),
                'age_at_date': self._age_at_date(date_str),
                'current_systolic_bp': self._current_systolic_bp(cur),
                'current_egfr': self._current_egfr(cur),
            })

            cur += timedelta(days=interval_days)

        return instances

    def is_high_risk_on_date(self, date: str, risk_window_days: int = 30) -> bool:
        """
        Determine if high-risk event occurs within future window.

        Parameters:
        -----------
        date : str
            Date to assess risk for (prediction start point)
        risk_window_days : int
            Days to look ahead for high-risk events (default 30)

        Returns:
        --------
        bool : True if high-risk event occurs in (date, date + window_days]
        """
        target_date = safe_parse_date(date)
        if not target_date:
            return False

        end_window = target_date + timedelta(days=risk_window_days)

        # Check risk events
        for risk_event in self.risk_events:
            event_date = safe_parse_date(risk_event['date'])
            if event_date and target_date < event_date <= end_window:
                return True

        # Check emergency visits
        for event in self.emergency_visit_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and target_date < event_date <= end_window:
                if event.get('diabetes_related', False):
                    return True

        # Check hospitalizations
        for event in self.hospitalization_timeline:
            event_date = safe_parse_date(event['date'])
            if event_date and target_date < event_date <= end_window:
                if event.get('diabetes_related', False):
                    return True

        # Check for very high HbA1c
        for reading in self.hba1c_timeline:
            reading_date = safe_parse_date(reading['date'])
            if reading_date and target_date < reading_date <= end_window:
                if reading.get('value', 0) >= 10.0:
                    return True

        return False

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

    @classmethod
    def get_feature_descriptions(cls) -> Dict[str, str]:
        """Get descriptions for all available features."""
        return {
            'age_at_date': 'Patient age on this date',
            'days_since_last_hba1c': 'Days since most recent HbA1c test',
            'days_since_last_encounter': 'Days since most recent medical encounter',
            'days_since_medication_change': 'Days since last medication change',
            'days_since_emergency_visit': 'Days since most recent emergency visit',
            'days_since_hospitalization': 'Days since most recent hospitalization',
            'encounters_last_90d': 'Number of medical encounters in last 90 days',
            'emergency_visits_last_180d': 'Number of emergency visits in last 180 days',
            'hospitalizations_last_365d': 'Number of hospitalizations in last year',
            'medication_changes_last_90d': 'Number of medication changes in last 90 days',
            'current_hba1c_level': 'Most recent HbA1c value (%)',
            'hba1c_trend_last_180d': 'HbA1c trend: -1=improving, 0=stable, 1=worsening',
            'diabetes_complication_count': 'Number of documented diabetes complications',
            'care_gaps_count': 'Number of care gaps (>6 months without HbA1c)',
            'current_systolic_bp': 'Most recent systolic blood pressure (mmHg)',
            'current_egfr': 'Most recent eGFR (mL/min/1.73m2)'
        }

    @classmethod
    def load_from_dataframes(cls,
                              patients_df: pd.DataFrame,
                              encounters_df: pd.DataFrame,
                              conditions_df: pd.DataFrame,
                              observations_df: pd.DataFrame,
                              medications_df: pd.DataFrame = None,
                              diabetic_only: bool = True,
                              show_progress: bool = True) -> Dict[str, 'ExtendedPatientProfile']:
        """
        Load ExtendedPatientProfile objects from DataFrames.

        This is the fast way to create profiles - loads all data once and
        populates timelines efficiently.

        Parameters:
        -----------
        patients_df : pd.DataFrame
            Patients table with ID, BIRTHDATE, GENDER columns
        encounters_df : pd.DataFrame
            Encounters table with PATIENT, DATE, DESCRIPTION, CODE columns
        conditions_df : pd.DataFrame
            Conditions table with PATIENT, DESCRIPTION columns
        observations_df : pd.DataFrame
            Observations table with PATIENT, DATE, DESCRIPTION, VALUE, CODE columns
        medications_df : pd.DataFrame, optional
            Medications table with PATIENT, START, DESCRIPTION columns
        diabetic_only : bool
            If True, only load diabetic patients (default: True)
        show_progress : bool
            If True, show progress bar (default: True)

        Returns:
        --------
        Dict[str, ExtendedPatientProfile] : Dictionary mapping patient_id to profile
        """
        def get_col(df, name):
            """Find column by name, case-insensitive."""
            name_lower = name.lower().replace('_', '')
            for col in df.columns:
                if col.lower().replace('_', '') == name_lower:
                    return col
            return None

        # Get column names
        patient_id_col = get_col(patients_df, 'id')
        birthdate_col = get_col(patients_df, 'birthdate')
        gender_col = get_col(patients_df, 'gender')

        enc_patient_col = get_col(encounters_df, 'patient')
        enc_date_col = get_col(encounters_df, 'date')
        enc_desc_col = get_col(encounters_df, 'description')
        enc_code_col = get_col(encounters_df, 'code')
        enc_class_col = get_col(encounters_df, 'encounter_class')

        cond_patient_col = get_col(conditions_df, 'patient')
        cond_desc_col = get_col(conditions_df, 'description')

        obs_patient_col = get_col(observations_df, 'patient')
        obs_date_col = get_col(observations_df, 'date')
        obs_desc_col = get_col(observations_df, 'description')
        obs_value_col = get_col(observations_df, 'value')
        obs_code_col = get_col(observations_df, 'code')

        # Find diabetic patients
        def is_diabetic_condition(desc):
            if pd.isna(desc):
                return False
            desc_lower = str(desc).lower()
            return any(term in desc_lower for term in DIABETES_TERMS)

        diabetic_patients = set(
            conditions_df[conditions_df[cond_desc_col].apply(is_diabetic_condition)][cond_patient_col].unique()
        )

        # Determine which patients to load
        if diabetic_only:
            patient_ids = diabetic_patients
        else:
            patient_ids = set(patients_df[patient_id_col].unique())

        print(f"Loading {len(patient_ids):,} patient profiles...")

        # Pre-parse dates in DataFrames (FAST - vectorized)
        encounters_df = encounters_df.copy()
        observations_df = observations_df.copy()
        encounters_df['_date_parsed'] = pd.to_datetime(encounters_df[enc_date_col], errors='coerce')
        observations_df['_date_parsed'] = pd.to_datetime(observations_df[obs_date_col], errors='coerce')

        # Pre-index all data by patient (convert to dict of lists for speed)
        enc_by_patient = {pid: grp.to_dict('records') for pid, grp in encounters_df.groupby(enc_patient_col)}
        obs_by_patient = {pid: grp.to_dict('records') for pid, grp in observations_df.groupby(obs_patient_col)}

        # Pre-index conditions by patient for complication extraction
        cond_start_col = get_col(conditions_df, 'start')
        cond_by_patient = {pid: grp.to_dict('records') for pid, grp in conditions_df.groupby(cond_patient_col)}

        # Diabetes complication terms (SNOMED descriptions from Synthea)
        complication_terms = [
            'neuropathy', 'retinopathy', 'nephropathy', 'microalbuminuria',
            'diabetic renal', 'macular edema', 'proteinuria',
            'blindness due to', 'amputation', 'gangrene', 'diabetic foot',
            'ketoacidosis', 'proliferative diabetic'
        ]

        if medications_df is not None:
            med_patient_col = get_col(medications_df, 'patient')
            med_start_col = get_col(medications_df, 'start')
            med_stop_col = get_col(medications_df, 'stop')
            med_desc_col = get_col(medications_df, 'description')
            med_by_patient = {pid: grp.to_dict('records') for pid, grp in medications_df.groupby(med_patient_col)}
        else:
            med_by_patient = {}
            med_stop_col = None

        # Create patient info lookup (drop duplicates to ensure unique index)
        patients_unique = patients_df.drop_duplicates(subset=[patient_id_col], keep='first')
        patient_info = patients_unique.set_index(patient_id_col).to_dict('index')

        # Build profiles
        profiles = {}
        iterator = tqdm(patient_ids, desc="Building profiles") if show_progress else patient_ids

        for patient_id in iterator:
            if patient_id not in patient_info:
                continue

            info = patient_info[patient_id]
            demographics = {
                'birthdate': info.get(birthdate_col, ''),
                'gender': info.get(gender_col, '')
            }

            profile = cls(patient_id, demographics)

            # Mark as diabetic if in diabetic_patients set
            if patient_id in diabetic_patients:
                profile.diabetes_profile['has_diabetes'] = True

            # Load diabetes complications from conditions
            if patient_id in cond_by_patient:
                for row in cond_by_patient[patient_id]:
                    desc = str(row.get(cond_desc_col, '')).lower()
                    if any(term in desc for term in complication_terms):
                        date = pd.to_datetime(row.get(cond_start_col), errors='coerce')
                        if pd.notna(date):
                            profile.complication_timeline.append({
                                'date': date,
                                'complication': str(row.get(cond_desc_col, '')),
                                'severity': 'unknown'
                            })
                if profile.complication_timeline:
                    profile.complication_timeline.sort(
                        key=lambda x: safe_parse_date(x['date']) or datetime.min
                    )

            # Load encounters into timeline (using pre-converted records)
            if patient_id in enc_by_patient:
                for row in enc_by_patient[patient_id]:
                    date = pd.to_datetime(row.get(enc_date_col), errors='coerce')
                    desc = str(row.get(enc_desc_col, ''))
                    enc_class = str(row.get(enc_class_col, '')).lower() if enc_class_col else ''

                    profile.encounter_timeline.append({
                        'date': date,
                        'encounter_type': desc,
                        'reason': desc,
                        'encounter_class': enc_class
                    })

                    # Check for ED visits or hospitalizations using ENCOUNTER_CLASS
                    if enc_class == 'emergency':
                        profile.emergency_visit_timeline.append({
                            'date': date,
                            'reason': desc,
                            'diabetes_related': True
                        })
                    elif enc_class == 'inpatient':
                        profile.hospitalization_timeline.append({
                            'date': date,
                            'reason': desc,
                            'diabetes_related': True
                        })

            # Load observations into timelines (using pre-converted records)
            if patient_id in obs_by_patient:
                for row in obs_by_patient[patient_id]:
                    date = pd.to_datetime(row.get(obs_date_col), errors='coerce')
                    desc = str(row.get(obs_desc_col, '')).lower()
                    value = row.get(obs_value_col)
                    code = str(row.get(obs_code_col, '')) if obs_code_col else ''

                    # HbA1c
                    if 'hba1c' in desc or 'hemoglobin a1c' in desc:
                        try:
                            profile.hba1c_timeline.append({
                                'date': date,
                                'value': float(value),
                                'units': '%'
                            })
                        except:
                            pass

                    # Systolic BP (LOINC 8480-6)
                    if 'systolic' in desc or code == '8480-6':
                        try:
                            profile.bp_timeline.append({
                                'date': date,
                                'systolic': float(value),
                                'diastolic': None
                            })
                        except:
                            pass

                    # eGFR (LOINC 33914-3)
                    if 'glomerular' in desc or 'egfr' in desc or code == '33914-3':
                        try:
                            profile.egfr_timeline.append({
                                'date': date,
                                'value': float(value)
                            })
                        except:
                            pass

                    # BMI (LOINC 39156-5)
                    if 'body mass' in desc or 'bmi' in desc or code == '39156-5':
                        try:
                            profile.bmi_timeline.append({
                                'date': date,
                                'value': float(value)
                            })
                        except:
                            pass

                    # Diastolic BP (LOINC 8462-4)
                    if 'diastolic' in desc or code == '8462-4':
                        try:
                            # Update diastolic in existing BP reading or add new
                            bp_date_match = [r for r in profile.bp_timeline if r.get('date') == date]
                            if bp_date_match:
                                bp_date_match[0]['diastolic'] = float(value)
                            else:
                                profile.bp_timeline.append({
                                    'date': date,
                                    'systolic': None,
                                    'diastolic': float(value)
                                })
                        except:
                            pass

            # Load medications (using pre-converted records)
            if patient_id in med_by_patient:
                for row in med_by_patient[patient_id]:
                    start_date = pd.to_datetime(row.get(med_start_col), errors='coerce')
                    desc = str(row.get(med_desc_col, ''))
                    profile.medication_timeline.append({
                        'date': start_date,
                        'action': 'started',
                        'medication': desc,
                        'dosage': ''
                    })
                    # Also load STOP date as a 'stopped' event
                    if med_stop_col:
                        stop_date = pd.to_datetime(row.get(med_stop_col), errors='coerce')
                        if pd.notna(stop_date):
                            profile.medication_timeline.append({
                                'date': stop_date,
                                'action': 'stopped',
                                'medication': desc,
                                'dosage': ''
                            })

            # Compute care gaps from HbA1c timeline
            if len(profile.hba1c_timeline) >= 2:
                sorted_readings = sorted(
                    profile.hba1c_timeline,
                    key=lambda x: safe_parse_date(x['date']) or datetime.min
                )
                for i in range(1, len(sorted_readings)):
                    prev_date = safe_parse_date(sorted_readings[i - 1]['date'])
                    curr_date = safe_parse_date(sorted_readings[i]['date'])
                    if prev_date and curr_date:
                        gap_days = (curr_date - prev_date).days
                        if gap_days > 180:
                            profile.care_gap_timeline.append({
                                'date': curr_date,
                                'gap_type': 'hba1c_monitoring',
                                'days_overdue': gap_days
                            })
                if profile.care_gap_timeline:
                    profile.care_gap_timeline.sort(
                        key=lambda x: safe_parse_date(x['date']) or datetime.min
                    )

            profiles[patient_id] = profile

        # Debug logging
        profiles_with_encounters = sum(1 for p in profiles.values() if p.encounter_timeline)
        profiles_60_days = 0
        for p in profiles.values():
            if p.encounter_timeline:
                dates = [e['date'] for e in p.encounter_timeline if e['date'] is not None]
                if len(dates) >= 2 and (max(dates) - min(dates)).days >= 60:
                    profiles_60_days += 1

        print(f"✅ Loaded {len(profiles):,} profiles")
        print(f"   With encounters: {profiles_with_encounters:,}")
        print(f"   With ≥60 days data: {profiles_60_days:,} (will generate instances)")
        return profiles

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
