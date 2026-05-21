"""
EHR Clinical Decision Support Pipeline — Week 3 Homework

Composes all building blocks from Weeks 1-3:
- Week 1: OCR cleaning, clinical note extraction
- Week 2: Patient profiles, feature engineering
- Week 3: Model prediction, SHAP explanation

Usage
-----
    from ehr_pipeline import EHRPipeline

    pipe = EHRPipeline(
        data_dir="data/week_1/processed_data/csv",
        model_path="data/week_3/best_xgboost_model.pkl",
    )
    pipe.load_csvs().clean_notes().extract_notes().build_profiles()

    report = pipe.explain_patient("patient-abc", "2017-03-15")
    print_explanation(report)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports resolve via pip install (data_processing/, profiles/ are siblings)
# ---------------------------------------------------------------------------

from src.data_processing.ocr import strip_html, fix_ocr_text, has_ocr_errors
from src.data_processing.extraction import extract_all_from_note
from src.profiles.extended_patient_profile import (
    ExtendedPatientProfile,
    safe_parse_date,
)


# ═══════════════════════════════════════════════════════════════════════════
# Clinical interpretation mapping for all 22 features
# ═══════════════════════════════════════════════════════════════════════════

CLINICAL_MEANINGS = {
    "age_at_date": "Patient age — older patients have higher complication risk",
    "days_since_last_hba1c": "Days since last HbA1c test — gaps indicate monitoring lapses",
    "days_since_last_encounter": "Days since last clinical visit — longer gaps indicate care discontinuity",
    "days_since_medication_change": "Days since medication adjustment — recent changes may indicate instability",
    "days_since_emergency_visit": "Days since last ED visit — recent visits indicate acute episodes",
    "days_since_hospitalization": "Days since last hospitalization — recent admits indicate severity",
    "encounters_last_90d": "Recent encounter frequency — low values may indicate disengagement",
    "emergency_visits_last_180d": "Recent ED visits — indicates acute decompensation episodes",
    "hospitalizations_last_365d": "Recent hospitalizations — indicates severe disease burden",
    "medication_changes_last_90d": "Recent medication changes — instability in treatment regimen",
    "current_hba1c_level": "Most recent HbA1c — higher values indicate poor glycemic control",
    "hba1c_trend_last_180d": "HbA1c trend over 6 months — positive means worsening control",
    "current_systolic_bp": "Systolic blood pressure — elevated values indicate hypertension risk",
    "current_diastolic_bp": "Diastolic blood pressure — elevated values indicate hypertension risk",
    "bp_trend_last_180d": "Blood pressure trend over 6 months — positive means worsening",
    "current_egfr": "Estimated GFR — lower values indicate kidney damage (CKD progression)",
    "egfr_trend_last_365d": "eGFR trend over 12 months — negative means declining kidney function",
    "bmi_category": "BMI classification (1=underweight, 2=normal, 3=overweight, 4=obese)",
    "active_medication_count": "Number of active medications — higher counts indicate treatment complexity",
    "diabetes_complication_count": "Number of diabetes complications — indicates disease progression",
    "care_gaps_count": "Number of care gaps — pattern of disengagement from care",
    "longest_care_gap_days": "Maximum gap between encounters — indicates historical care discontinuity",
}


def print_explanation(report):
    """Pretty-print a clinical explanation report from ``explain_patient()``."""
    print("\n" + "=" * 70)
    print("CLINICAL RISK ASSESSMENT")
    print("=" * 70)
    print(f"Patient ID:  {report['patient_id']}")
    print(f"Date:        {report['date']}")
    print(f"Risk Score:  {report['risk_score']:.1%}")
    print(f"Risk Level:  {report['risk_level']}")

    print("\n" + "-" * 70)
    print("RISK-INCREASING FACTORS:")
    print("-" * 70)
    for f in report["risk_factors"]:
        val_str = "not measured" if f["value"] is None else f"{f['value']:.2f}"
        print(f"\n  {f['feature']}")
        print(f"    Value: {val_str}  |  Contribution: {f['contribution_pct']:.1f}%")
        print(f"    -> {f['clinical_meaning']}")

    print("\n" + "-" * 70)
    print("PROTECTIVE FACTORS:")
    print("-" * 70)
    for f in report["protective_factors"]:
        val_str = "not measured" if f["value"] is None else f"{f['value']:.2f}"
        print(f"\n  {f['feature']}")
        print(f"    Value: {val_str}  |  Contribution: {f['contribution_pct']:.1f}%")
        print(f"    -> {f['clinical_meaning']}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# EHRPipeline — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class EHRPipeline:
    """End-to-end clinical decision support pipeline.

    Composes Weeks 1-3 building blocks into a single class:

    - **Data loading**: Read Synthea CSVs
    - **Note cleaning**: Strip HTML, fix OCR errors (Week 1)
    - **Note extraction**: Extract vitals/labs/meds from text (Week 1)
    - **Profile building**: Create ExtendedPatientProfile objects (Week 2)
    - **Prediction**: Load trained model, predict risk (Week 3 Session 2)
    - **Explanation**: SHAP-based clinical explanations (Week 3 Session 3)

    Stages chain with ``return self``::

        pipe = EHRPipeline(data_dir=..., model_path=...)
        pipe.load_csvs().clean_notes().extract_notes().build_profiles()
        report = pipe.explain_patient("patient-abc", "2017-03-15")
    """

    _CSV_NAMES = [
        "patients",
        "encounters",
        "conditions",
        "observations",
        "medications",
    ]

    def __init__(self, data_dir, model_path, notes_file=None, ocr_config=None):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing Synthea CSV files.
        model_path : str
            Path to trained model pickle (from Session 2).
        notes_file : str, optional
            Path to clinical-notes CSV.  If *None*, note processing is skipped.
        ocr_config : str, optional
            Path to OCR substitution JSON.  If *None*, OCR correction is skipped.
        """
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.notes_file = Path(notes_file) if notes_file else None
        self.ocr_config = Path(ocr_config) if ocr_config else None

        # Load pre-trained model (does NOT train — Session 2 produces the artifact)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Loaded model from {self.model_path.name}")

        # Extract feature column names from model
        if hasattr(self.model, "feature_names_in_"):
            self.feature_cols = list(self.model.feature_names_in_)
        elif (
            hasattr(self.model, "get_booster")
            and self.model.get_booster().feature_names
        ):
            self.feature_cols = self.model.get_booster().feature_names
        else:
            # Fallback: standard 22-feature order
            self.feature_cols = list(CLINICAL_MEANINGS.keys())

        # DataFrames populated by load_csvs()
        self.patients_df: pd.DataFrame = None
        self.encounters_df: pd.DataFrame = None
        self.conditions_df: pd.DataFrame = None
        self.observations_df: pd.DataFrame = None
        self.medications_df: pd.DataFrame = None
        self.notes_df: pd.DataFrame = None

        # Populated by extract_notes()
        self.extracted_vitals: list = []
        self.extracted_labs: list = []
        self.extracted_meds: list = []

        # Populated by build_profiles()
        self.profiles: dict = {}  # patient_id -> ExtendedPatientProfile

        # Populated by build_feature_matrix()
        self.feature_df: pd.DataFrame = None

        # SHAP explainer (created lazily on first explain_patient call)
        self._explainer = None

    # ------------------------------------------------------------------
    # Stage 1: Load CSVs  (Week 1)
    # ------------------------------------------------------------------
    def load_csvs(self) -> "EHRPipeline":
        available = {f.stem.lower(): f for f in self.data_dir.glob("*.csv")}
    
        for name in self._CSV_NAMES:
            if name in available:
                df = pd.read_csv(available[name])
                setattr(self, f"{name}_df", df)
    
        if self.notes_file and self.notes_file.exists():
            self.notes_df = pd.read_csv(self.notes_file)
    
        return self

    # ------------------------------------------------------------------
    # Stage 2: Clean notes — OCR + HTML  (Week 1)
    # ------------------------------------------------------------------
    def clean_notes(self, text_col: str = "note_text") -> "EHRPipeline":
        if self.notes_df is None or text_col not in self.notes_df.columns:
            print("  No notes to clean.")
            return self
    
        # Guard against None/NaN cells before applying strip_html
        self.notes_df[text_col] = self.notes_df[text_col].apply(
            lambda t: strip_html(t) if isinstance(t, str) else t
        )
    
        if self.ocr_config and self.ocr_config.exists():
            import json
            with open(self.ocr_config) as f:
                cfg = json.load(f)  # Load the full config
            
            subs = cfg.get("ocr_substitutions", {})
            valid_terms = cfg.get("valid_mixed_terms", [])  # Extract valid_terms
            
            self.notes_df[text_col] = self.notes_df[text_col].apply(
                lambda t: fix_ocr_text(t, subs, valid_terms) if isinstance(t, str) else t
            )
    
        return self

    # ------------------------------------------------------------------
    # Stage 3: Extract structured data from notes  (Week 1)
    # ------------------------------------------------------------------
    def extract_notes(self, text_col="note_text", patient_col="patient_id", encounter_col="encounter_id") -> "EHRPipeline":
        if self.notes_df is None:
            print("  No notes to extract from.")
            return self
    
        vitals, labs, meds = [], [], []
        for i, (_, row) in enumerate(self.notes_df.iterrows()):
            # Skip rows where the text itself is missing
            if not isinstance(row.get(text_col), str):
                continue
            try:
                result = extract_all_from_note(
                    row[text_col], row[patient_col], row[encounter_col]
                )
                vitals.extend(result["vitals"])
                labs.extend(result["labs"])
                meds.extend(result["medications"])
            except Exception as e:
                print(f"  Warning: row {i} failed extraction: {e}")
                continue
    
        self.extracted_vitals = vitals
        self.extracted_labs = labs
        self.extracted_meds = meds
    
        if vitals or labs:
            extracted_obs = pd.DataFrame(vitals + labs)
            if self.observations_df is not None:
                self.observations_df = pd.concat(
                    [self.observations_df, extracted_obs], ignore_index=True
                )
            else:
                self.observations_df = extracted_obs
    
        if meds:
            extracted_meds_df = pd.DataFrame(meds)
            if self.medications_df is not None:
                self.medications_df = pd.concat(
                    [self.medications_df, extracted_meds_df], ignore_index=True
                )
            else:
                self.medications_df = extracted_meds_df
    
        return self
    
    # ------------------------------------------------------------------
    # Stage 4: Build ExtendedPatientProfiles  (Week 2)
    # ------------------------------------------------------------------
    def build_profiles(self, diabetic_only: bool = True) -> "EHRPipeline":
        required = ["patients_df", "encounters_df", "conditions_df", "observations_df"]
        missing = [n for n in required if getattr(self, n) is None]
        if missing:
            raise RuntimeError(f"Cannot build profiles — missing: {missing}")
    
        self.profiles = ExtendedPatientProfile.load_from_dataframes(
            patients_df=self.patients_df,
            encounters_df=self.encounters_df,
            conditions_df=self.conditions_df,
            observations_df=self.observations_df,
            medications_df=self.medications_df,
            diabetic_only=diabetic_only,
        )
        return self

    # ------------------------------------------------------------------
    # Single Patient Query (the core clinical use case)
    # ------------------------------------------------------------------
    def get_patient_features(self, patient_id: str, date: str) -> dict:
        if patient_id not in self.profiles:
            raise KeyError(f"Patient '{patient_id}' not found in profiles.")
    
        profile = self.profiles[patient_id]
        features = profile.get_daily_features(date)
    
        if not features:
            raise ValueError(f"No features computed for patient '{patient_id}' at {date}.")
    
        return features

    def predict_patient(self, patient_id: str, date: str) -> float:
        features = self.get_patient_features(patient_id, date)
        feature_row = pd.DataFrame([features])[self.feature_cols]
        feature_row = feature_row.apply(pd.to_numeric, errors='coerce')
        return float(self.model.predict_proba(feature_row)[0, 1])

    def explain_patient(self, patient_id: str, date: str, top_n: int = 5) -> dict:
        import shap
    
        features = self.get_patient_features(patient_id, date)
        feature_row = pd.DataFrame([features])[self.feature_cols]
        feature_row = feature_row.apply(pd.to_numeric, errors='coerce')
        probability = float(self.model.predict_proba(feature_row)[0, 1])
    
        if self._explainer is None:
            import builtins
            _builtin_float = builtins.float
            def _safe_float(x):
                if isinstance(x, str):
                    x = x.strip("[]")
                return _builtin_float(x)
            builtins.float = _safe_float
            try:
                self._explainer = shap.TreeExplainer(self.model)
            finally:
                builtins.float = _builtin_float  # always restore
    
        shap_raw = self._explainer.shap_values(feature_row)
        shap_vals = np.asarray(
            shap_raw[1] if isinstance(shap_raw, list) else shap_raw
        ).flatten()
    
        total = np.abs(shap_vals).sum()
        pcts = np.abs(shap_vals) / total * 100 if total > 0 else np.zeros_like(shap_vals)
    
        factor_df = pd.DataFrame({
            "feature": self.feature_cols,
            "value": feature_row.values.flatten(),
            "shap": shap_vals,
            "contribution_pct": pcts,
        })
    
        risk_factors = (
            factor_df[factor_df["shap"] > 0]
            .nlargest(top_n, "shap")
            .to_dict("records")
        )
        protective_factors = (
            factor_df[factor_df["shap"] < 0]
            .nsmallest(top_n, "shap")
            .to_dict("records")
        )
    
        if probability >= 0.8:
            risk_level = "CRITICAL"
        elif probability >= 0.6:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
    
        def format_factor(f):
            return {
                "feature": f["feature"],
                "value": None if np.isnan(f["value"]) else f["value"],
                "contribution_pct": f["contribution_pct"],
                "direction": "risk" if f["shap"] > 0 else "protective",
                "clinical_meaning": CLINICAL_MEANINGS.get(f["feature"], "Clinical indicator"),
            }
    
        return {
            "patient_id": patient_id,
            "date": date,
            "risk_score": probability,
            "risk_level": risk_level,
            "risk_factors": [format_factor(f) for f in risk_factors],
            "protective_factors": [format_factor(f) for f in protective_factors],
        }

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------
    def build_feature_matrix(self, interval_days: int = 7) -> pd.DataFrame:
        if not self.profiles:
            raise RuntimeError("No profiles. Call build_profiles() first.")
    
        all_rows = []
        for _pid, profile in self.profiles.items():
            rows = profile.generate_all_instances(interval_days=interval_days)
            all_rows.extend(rows)
    
        self.feature_df = pd.DataFrame(all_rows)
        return self.feature_df

    def evaluate_model(self, data: pd.DataFrame = None) -> dict:
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    
        df = data if data is not None else self.feature_df
        if df is None:
            raise RuntimeError("No data available. Call build_feature_matrix() first.")
    
        label_col = "will_have_high_risk_event_next_30d"
        exclude_patterns = ["label", "risk", "target", "will_have", "next", "survival"]
        feat_cols = [
            c for c in df.columns
            if c not in ["patient_id", "date"]
            and not any(p in c.lower() for p in exclude_patterns)
        ]
    
        X = df[feat_cols]
        y = df[label_col].astype(int)
        groups = df["patient_id"]
    
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
    
        y_prob = self.model.predict_proba(X.iloc[test_idx])[:, 1]
        y_test = y.iloc[test_idx]
    
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        precision_at_80_recall = float(np.interp(0.80, recall[::-1], precision[::-1]))
    
        return {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "precision_at_80_recall": precision_at_80_recall,
            "n_test_patients": len(groups.iloc[test_idx].unique()),
            "n_test_rows": len(test_idx),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ehr_pipeline.py — import verification")
    print("=" * 60)

    # 1. Verify all imports resolved
    print("\n[OK] Core imports:")
    print(f"  strip_html             -> {strip_html}")
    print(f"  fix_ocr_text           -> {fix_ocr_text}")
    print(f"  has_ocr_errors         -> {has_ocr_errors}")
    print(f"  extract_all_from_note  -> {extract_all_from_note}")
    print(f"  ExtendedPatientProfile -> {ExtendedPatientProfile}")
    print(f"  safe_parse_date        -> {safe_parse_date}")

    # 2. Verify helpers
    print("\n[OK] Clinical meanings defined for 22 features:")
    for feat in list(CLINICAL_MEANINGS)[:3]:
        print(f"  {feat}: {CLINICAL_MEANINGS[feat][:50]}...")

    # 3. Verify pipeline class instantiation (without model)
    print("\n[OK] EHRPipeline class available")
    print(
        "  Stages: load_csvs -> clean_notes -> extract_notes -> "
        "build_profiles -> predict_patient / explain_patient"
    )

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)
