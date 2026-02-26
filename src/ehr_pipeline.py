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

from data_processing.ocr import strip_html, fix_ocr_text, has_ocr_errors
from data_processing.extraction import extract_all_from_note
from profiles.extended_patient_profile import (
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
        """Read Synthea CSVs from ``self.data_dir`` into DataFrames.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. Use ``self.data_dir.glob("*.csv")`` to list available files.
             Build a dict mapping lowercase stem -> Path.
          2. Loop over ``self._CSV_NAMES``. For each name, check if a
             matching file exists. If so, read it with ``pd.read_csv()``
             and store via ``setattr(self, f"{name}_df", df)``.
          3. If ``self.notes_file`` exists, read it into ``self.notes_df``.
          4. Return ``self`` for chaining.
        """
        # TODO: Build dict of available CSV files
        # available = {f.stem.lower(): f for f in self.data_dir.glob("*.csv")}

        # TODO: Loop over self._CSV_NAMES and load each
        # for name in self._CSV_NAMES:
        #     ...

        # TODO: Load notes file if provided
        # if self.notes_file and self.notes_file.exists():
        #     ...

        raise NotImplementedError("TODO: implement load_csvs()")

    # ------------------------------------------------------------------
    # Stage 2: Clean notes — OCR + HTML  (Week 1)
    # ------------------------------------------------------------------
    def clean_notes(self, text_col: str = "note_text") -> "EHRPipeline":
        """Strip HTML and fix OCR errors in ``self.notes_df``.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. If ``self.notes_df`` is None or ``text_col`` not in columns,
             print a message and return self.
          2. Apply ``strip_html()`` to the text column.
          3. If ``self.ocr_config`` exists, load it as JSON and apply
             ``fix_ocr_text()`` using the substitution dict.
          4. Return ``self`` for chaining.

        Hint: ``strip_html`` and ``fix_ocr_text`` are already imported
        at the top of this file.
        """
        # TODO: Check if notes exist
        # if self.notes_df is None or text_col not in self.notes_df.columns:
        #     print("  No notes to clean.")
        #     return self

        # TODO: Apply strip_html to each note
        # self.notes_df[text_col] = self.notes_df[text_col].apply(strip_html)

        # TODO: Apply OCR correction if config provided
        # if self.ocr_config and self.ocr_config.exists():
        #     import json
        #     ...

        raise NotImplementedError("TODO: implement clean_notes()")

    # ------------------------------------------------------------------
    # Stage 3: Extract structured data from notes  (Week 1)
    # ------------------------------------------------------------------
    def extract_notes(
        self,
        text_col: str = "note_text",
        patient_col: str = "patient_id",
        encounter_col: str = "encounter_id",
    ) -> "EHRPipeline":
        """Run NLP extraction on each note and merge results.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. If ``self.notes_df`` is None, print a message and return self.
          2. Loop over rows. For each note, call
             ``extract_all_from_note(text, patient_id, encounter_id)``.
          3. Collect returned vitals, labs, medications into lists.
          4. Store as ``self.extracted_vitals``, etc.
          5. Convert to DataFrames and merge into ``self.observations_df``
             and ``self.medications_df`` using ``pd.concat()``.
          6. Return ``self`` for chaining.

        Hint: ``extract_all_from_note`` returns a dict with keys
        'vitals', 'labs', 'medications' — each is a list of dicts.
        """
        # TODO: Check if notes exist
        # if self.notes_df is None:
        #     print("  No notes to extract from.")
        #     return self

        # TODO: Loop over notes and extract
        # vitals, labs, meds = [], [], []
        # for _, row in self.notes_df.iterrows():
        #     result = extract_all_from_note(...)
        #     vitals.extend(result["vitals"])
        #     ...

        # TODO: Merge extracted observations into self.observations_df
        # TODO: Merge extracted medications into self.medications_df

        raise NotImplementedError("TODO: implement extract_notes()")

    # ------------------------------------------------------------------
    # Stage 4: Build ExtendedPatientProfiles  (Week 2)
    # ------------------------------------------------------------------
    def build_profiles(self, diabetic_only: bool = True) -> "EHRPipeline":
        """Delegate to ``ExtendedPatientProfile.load_from_dataframes()``.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. Check that required DataFrames (patients, encounters,
             conditions, observations) are loaded. Raise RuntimeError
             if any are missing.
          2. Call ``ExtendedPatientProfile.load_from_dataframes()``
             with all available DataFrames.
          3. Store the result in ``self.profiles``.
          4. Return ``self`` for chaining.
        """
        # TODO: Check that required DataFrames exist
        # required = ["patients_df", "encounters_df", "conditions_df", "observations_df"]
        # missing = [n for n in required if getattr(self, n) is None]
        # if missing:
        #     raise RuntimeError(f"Cannot build profiles — missing: {missing}")

        # TODO: Call ExtendedPatientProfile.load_from_dataframes(...)
        # self.profiles = ExtendedPatientProfile.load_from_dataframes(
        #     patients_df=..., encounters_df=..., conditions_df=...,
        #     observations_df=..., medications_df=..., diabetic_only=diabetic_only,
        # )

        raise NotImplementedError("TODO: implement build_profiles()")

    # ------------------------------------------------------------------
    # Single Patient Query (the core clinical use case)
    # ------------------------------------------------------------------
    def get_patient_features(self, patient_id: str, date: str) -> dict:
        """Compute 22 features for a patient at a given cutoff date.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. Check that ``patient_id`` exists in ``self.profiles``.
             Raise KeyError if not.
          2. Get the profile: ``self.profiles[patient_id]``.
          3. Call ``profile.get_daily_features(date)`` to compute
             the 22 temporal features at the given cutoff.
          4. If features are empty, raise ValueError.
          5. Return the features dict.

        Parameters
        ----------
        patient_id : str
        date : str  (format ``"YYYY-MM-DD"``)

        Returns
        -------
        dict : feature_name -> value  (NaN for missing measurements)
        """
        # TODO: Look up the patient's profile
        # TODO: Call get_daily_features(date)
        # TODO: Return the features dict

        raise NotImplementedError("TODO: implement get_patient_features()")

    def predict_patient(self, patient_id: str, date: str) -> float:
        """Predict 30-day high-risk event probability for a patient.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. Call ``self.get_patient_features(patient_id, date)``.
          2. Build a 1-row DataFrame with columns in ``self.feature_cols``
             order (the model expects this exact column order).
          3. Call ``self.model.predict_proba(feature_row)[0, 1]`` to
             get the positive-class probability.
          4. Return as float.

        Note: NaN passthrough — XGBoost handles missing values natively,
        so no imputation is needed.
        """
        # TODO: Get features
        # features = self.get_patient_features(patient_id, date)

        # TODO: Build DataFrame in correct column order
        # feature_row = pd.DataFrame([features])[self.feature_cols]

        # TODO: Predict and return probability
        # probability = self.model.predict_proba(feature_row)[0, 1]

        raise NotImplementedError("TODO: implement predict_patient()")

    def explain_patient(
        self, patient_id: str, date: str, top_n: int = 5
    ) -> dict:
        """Generate SHAP-based clinical explanation for a patient's risk.

        HOMEWORK TODO: Implement this method.

        Uses TreeExplainer to identify contributing factors.  Returns a
        structured dict with:

        - ``risk_score``: probability 0-1
        - ``risk_level``: CRITICAL / HIGH / MODERATE / LOW
        - ``risk_factors``: list of top factors increasing risk
        - ``protective_factors``: list of top factors decreasing risk

        Each factor has ``contribution_pct`` and ``clinical_meaning``.
        No raw SHAP log-odds values are exposed.

        Steps:
          1. Get features and build feature_row (same as predict_patient).
          2. Get probability from model.predict_proba.
          3. Create SHAP TreeExplainer lazily (store in self._explainer).
             IMPORTANT: SHAP 0.49 with XGBoost >= 2.0 stores base_score
             as '[5E-1]' which causes ValueError.  Monkey-patch
             builtins.float temporarily (see Session 3 for pattern).
          4. Compute SHAP values: explainer.shap_values(feature_row).
             Handle list vs array return format.
          5. Convert SHAP to contribution_pct: |shap_i| / sum(|shap|) * 100.
          6. Split into risk_factors (shap > 0) and protective_factors
             (shap < 0), take top_n of each.
          7. Determine risk_level from probability thresholds
             (CRITICAL >= 0.8, HIGH >= 0.6, MODERATE >= 0.4, else LOW).
          8. Format each factor as a dict with feature, value,
             contribution_pct, direction, clinical_meaning.
          9. Return the full report dict.

        Hint: Use CLINICAL_MEANINGS.get(feature, "Clinical indicator...")
        for the clinical_meaning field.
        """
        import shap

        # TODO: Get features and prediction
        # features = self.get_patient_features(patient_id, date)
        # feature_row = pd.DataFrame([features])[self.feature_cols]
        # probability = self.model.predict_proba(feature_row)[0, 1]

        # TODO: Create SHAP explainer lazily (with builtins.float fix)
        # if self._explainer is None:
        #     import builtins
        #     _builtin_float = builtins.float
        #     def _safe_float(x):
        #         ...  # strip '[]' from string inputs
        #     builtins.float = _safe_float
        #     self._explainer = shap.TreeExplainer(self.model)
        #     builtins.float = _builtin_float  # always restore immediately

        # TODO: Compute SHAP values
        # shap_raw = self._explainer.shap_values(feature_row)
        # Handle list vs array return:
        #   if isinstance(shap_raw, list): use shap_raw[1]
        #   else: use shap_raw
        # shap_vals = np.asarray(...).flatten()

        # TODO: Convert to contribution_pct
        # total = np.abs(shap_vals).sum()
        # pcts = np.abs(shap_vals) / total * 100 if total > 0 else ...

        # TODO: Build factor DataFrame with columns:
        #   feature, value, shap, contribution_pct
        # Split into risk_factors (shap > 0, nlargest) and
        # protective_factors (shap < 0, nsmallest)

        # TODO: Determine risk_level

        # TODO: Format factors as list of dicts with:
        #   feature, value (None if NaN), contribution_pct,
        #   direction, clinical_meaning

        # TODO: Return report dict with:
        #   patient_id, date, risk_score, risk_level,
        #   risk_factors, protective_factors

        raise NotImplementedError("TODO: implement explain_patient()")

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------
    def build_feature_matrix(self, interval_days: int = 7) -> pd.DataFrame:
        """Generate a classifier-ready DataFrame with 22+ features.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. Check that ``self.profiles`` is not empty.
             Raise RuntimeError if so.
          2. Loop over all profiles. For each, call
             ``profile.generate_all_instances(interval_days=interval_days)``.
          3. Collect all rows and build a single DataFrame.
          4. Store as ``self.feature_df`` and return it.
        """
        # TODO: Check profiles exist
        # if not self.profiles:
        #     raise RuntimeError("No profiles. Call build_profiles() first.")

        # TODO: Loop over profiles and generate instances
        # all_rows = []
        # for _pid, profile in self.profiles.items():
        #     rows = profile.generate_all_instances(interval_days=interval_days)
        #     all_rows.extend(rows)

        # TODO: Build DataFrame and store
        # self.feature_df = pd.DataFrame(all_rows)

        raise NotImplementedError("TODO: implement build_feature_matrix()")

    def evaluate_model(self, data: pd.DataFrame = None) -> dict:
        """Evaluate the loaded model on patient data.

        HOMEWORK TODO: Implement this method.

        Steps:
          1. Use ``data`` if provided, else ``self.feature_df``.
             Raise RuntimeError if neither is available.
          2. Identify the label column:
             ``"will_have_high_risk_event_next_30d"``.
          3. Identify feature columns: all columns except patient_id,
             date, and anything containing label/risk/target/will_have/
             next/survival patterns.
          4. Split by patient using GroupShuffleSplit(test_size=0.2,
             random_state=42) with patient_id as groups.
          5. Predict probabilities on test set.
          6. Compute: roc_auc_score, average_precision_score,
             precision at 80% recall (interpolated from PR curve).
          7. Return results dict.

        Parameters
        ----------
        data : DataFrame, optional
            Pre-computed feature matrix (e.g. loaded from CSV).
            If *None*, uses ``self.feature_df``.

        Returns
        -------
        dict with roc_auc, pr_auc, precision_at_80_recall, etc.
        """
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            precision_recall_curve,
        )

        # TODO: Get data (from argument or self.feature_df)
        # df = data if data is not None else self.feature_df
        # if df is None:
        #     raise RuntimeError("No data available.")

        # TODO: Define label_col and identify feature columns
        # label_col = "will_have_high_risk_event_next_30d"
        # exclude_patterns = ["label", "risk", "target", "will_have", "next", "survival"]
        # feat_cols = [c for c in df.columns if c not in ["patient_id", "date"]
        #              and not any(p in c.lower() for p in exclude_patterns)]

        # TODO: Split by patient using GroupShuffleSplit
        # X = df[feat_cols]; y = df[label_col].astype(int); groups = df["patient_id"]
        # gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        # train_idx, test_idx = next(gss.split(X, y, groups))

        # TODO: Predict on test set
        # y_prob = self.model.predict_proba(X.iloc[test_idx])[:, 1]

        # TODO: Compute metrics (roc_auc, pr_auc, precision at 80% recall)
        # Hint: use np.interp(0.80, recall[::-1], precision[::-1]) for interpolation

        # TODO: Return results dict

        raise NotImplementedError("TODO: implement evaluate_model()")


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
