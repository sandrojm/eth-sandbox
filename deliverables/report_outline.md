# Project Report Outline
## Patient Monitoring via Structured Electronic Health Records
### SwissCare Health Network AG — AI Project Course

---

# Executive Summary

SwissCare Health Network AG processes over 3 million patient encounters annually, yet 70% of clinical information is locked in unstructured notes — creating information gaps that drive preventable hospital readmissions and expose the organisation to regulatory risk under the EU AI Act and GDPR. This project delivers an end-to-end ML pipeline that transforms raw EHR data into standardised patient profiles and predicts 30-day high-risk adverse events in diabetic patients, targeting a 48–72 hour intervention window where clinical action has the highest impact. The pipeline comprises four stages: clinical note cleaning and structured extraction, FHIR-inspired patient profile construction with 22 temporally-safe features, an XGBoost risk classifier selected for its superior PR-AUC over five baselines (including a single-feature HbA1c rule at PR-AUC ~0.155), and an autonomous monitoring system with configurable alert thresholds and a SHAP-based explanation layer that surfaces the top risk factors per patient in plain clinical language. The final model achieves [ROC-AUC] / [PR-AUC] on a patient-held-out test set and was validated for demographic fairness, calibration, and temporal generalisation. An explored 25-feature variant incorporating ZIP-code census data (population density, provider density, insurance coverage) found near-zero correlations with 30-day outcomes and is not recommended for deployment. Automated scoring reduces the manual review burden to flagged high-risk alerts only, with each prevented readmission avoiding an estimated [X CHF] in acute care costs; regulatory compliance documentation produced alongside the model directly mitigates penalty exposure under EU AI Act high-risk provisions. We recommend deploying the 22-feature model, piloting alert thresholds in a single ward before hospital-wide rollout to calibrate the precision-recall balance acceptable to nursing staff, and treating human oversight mechanisms and audit logging as first-class engineering requirements from the outset of any similar initiative.

---

# 1. Introduction & System Overview

## 1.1 Business Context and Problem Statement
- SwissCare Health Network manages hospitals and clinics across Switzerland and the EU, processing over 3 million patient encounters annually
- 70% of clinical information lives in unstructured notes, making it inaccessible for systematic analysis or real-time risk scoring
- Hospital readmissions result from information gaps between providers; manual review at the scale needed is not feasible with current staffing
- Regulatory landscape (EU AI Act, GDPR/revFADP, Swiss healthcare law) requires AI systems used in clinical decision support to be transparent, auditable, and human-supervised
- The compounding effect: fragmented data → missed early warning signals → adverse events → regulatory exposure

## 1.2 Project Objectives
- Automate the conversion of unstructured clinical notes and structured EHR tables into standardized patient profiles
- Enable proactive risk identification at least 48–72 hours before adverse events, without requiring extensive manual review
- Achieve full compliance with EU AI Act (high-risk classification), GDPR/revFADP Article 22 (automated decisions), HL7 FHIR (data interoperability), and SNOMED CT/ICD-10 (clinical coding)

## 1.3 Scope and Dataset
- Synthea-generated EHR dataset with 50,000+ patient records across 7 relational tables: patients, encounters, conditions, medications, observations, procedures, allergies
- Approximately 25% of encounters exist only as unstructured clinical notes (free text, semi-structured reports)
- Focus population: diabetic patients, as identified by diagnosis codes and clinical terminology matching
- Prediction target: 30-day high-risk adverse event (emergency visits, hospitalizations, acute complications, HbA1c ≥ 10.0%)

## 1.4 Stakeholders and Success Metrics
- Chief Medical Officer: adverse event prediction rate, false positive rate
- Emergency Department Director: alert recall and precision
- Nursing Leadership: alert interpretability, false positive alert rate (alert fatigue)
- Chief Information Officer: system reliability, FHIR formatting compliance
- Compliance Officer: zero regulatory violations, documentation completeness
- Patients: outcome improvement, meaningful information about automated decisions affecting their care

---

# 2. Technical Decision Justification

## 2.1 Data Variables and Privacy Justification
- Structured tables used: patient demographics (age, gender, race), encounter records (dates, types, descriptions), diagnosis conditions (ICD-10 codes), medication records (names, start/stop dates), lab observations (HbA1c, eGFR, blood pressure, glucose, creatinine)
- Unstructured source: clinical notes containing vitals, lab values, and medication mentions not captured in structured tables
- Justification for using personal health data: all fields are clinically necessary for risk prediction; no social identifiers (names, addresses, insurance numbers) are used as model features; data is pseudonymised by patient UUID; HbA1c and eGFR are direct clinical indicators of diabetic disease progression and are standard-of-care monitoring values; demographic variables (age, gender) are used for fairness assessment, not as primary predictors
- Variables excluded from modeling: patient name, address, exact birthdate (replaced by age), insurance fields — excluded as they serve no clinical predictive function and carry unnecessary privacy risk

## 2.2 Feature Engineering: The 22-Feature Set

### 2.2.1 Temporal Counter Features
- days_since_last_hba1c: gaps indicate monitoring lapses and disengagement from care
- days_since_last_encounter: care discontinuity signal
- days_since_medication_change: recent changes may indicate treatment instability
- days_since_emergency_visit: recent ED visits indicate acute decompensation
- days_since_hospitalization: recent admits indicate severe disease burden

### 2.2.2 Rolling Window Features
- encounters_last_90d: low values may indicate patient disengagement from care
- emergency_visits_last_180d: frequency of acute decompensation episodes
- hospitalizations_last_365d: cumulative disease severity
- medication_changes_last_90d: treatment regimen instability

### 2.2.3 Clinical Value Features
- current_hba1c_level: most recent HbA1c; higher values indicate poor glycemic control
- current_systolic_bp / current_diastolic_bp: hypertension as comorbidity risk
- current_egfr: lower values indicate chronic kidney disease progression
- bmi_category: obesity as a compounding risk factor (encoded as ordinal: 1=underweight, 2=normal, 3=overweight, 4=obese)

### 2.2.4 Trend Features (6-month / 12-month windows)
- hba1c_trend_last_180d: worsening vs. improving glycemic control over time
- bp_trend_last_180d: cardiovascular risk trajectory
- egfr_trend_last_365d: kidney function decline rate

### 2.2.5 Composite Risk Features
- active_medication_count: treatment complexity indicator
- diabetes_complication_count: cumulative disease progression score
- care_gaps_count: pattern of disengagement from care system
- longest_care_gap_days: historical care discontinuity, captures chronic non-adherence
- age_at_date: older patients carry elevated baseline complication risk

### 2.2.6 Temporal Safety Design
- All features use a cutoff_time parameter: only data strictly before the prediction date is used
- This prevents data leakage — a common and critical error in healthcare ML where future observations contaminate training
- Patient-level train/test splitting using GroupShuffleSplit ensures no patient appears in both train and test sets; row-level random splitting would allow correlated observations from the same patient to appear on both sides, inflating reported performance

## 2.3 Evaluation Metrics Selection and Stakeholder Alignment
- Primary metric: PR-AUC (Area Under the Precision-Recall Curve) — chosen because the dataset has severe class imbalance (~3% positive rate); ROC-AUC is optimistic under imbalance because it accounts for true negatives, which are abundant and easy to predict correctly
- Secondary metric: ROC-AUC — reported alongside PR-AUC for comparability with published literature and benchmarks
- Clinical threshold metric: Precision at 80% Recall — directly answers the clinical question "if we catch 80% of high-risk patients, how many false alarms do we generate per true alert?"
- Rationale for 80% recall threshold: reflects ED Director and Nursing Leadership priorities; missing 20% of high-risk patients is acceptable if it meaningfully reduces alert fatigue on nursing floors
- Stakeholder mapping: CMO and ED Director prioritize recall (catching events); Nursing Leadership prioritizes precision (alert quality); Compliance Officer prioritizes calibration and documentation

## 2.4 Model Architecture Decisions

### 2.4.1 Baseline Models (Week 2)
- Random prediction: establishes absolute floor (coin flip biased to class distribution)
- Majority class prediction: demonstrates why accuracy is misleading under imbalance — predicting "low risk" for every patient gives ~97% accuracy but zero clinical utility
- Single-feature HbA1c threshold (> 9%): tests whether a single clinical rule matches ML; establishes clinical domain baseline; PR-AUC ~0.155
- Risk factor counting: combines multiple binary indicators; tests whether feature combination helps without learned weights
- Logistic Regression with class_weight='balanced': first learned model; upper baseline for linear approaches; interpretable coefficients

### 2.4.2 Advanced Models (Week 3)
- Decision Tree: provides full decision path interpretability; prone to overfitting, included for transparency comparison
- Random Forest: ensemble via bagging; reduces variance over single tree; robust to feature scale differences
- XGBoost (final model): sequential boosting; handles NaN natively (no imputation needed for missing labs); optimized with scale_pos_weight for class imbalance; eval_metric='aucpr' to optimize directly for the primary metric; hyperparameter search via random search
- LSTM (temporal sequence model): explored as an alternative to tabular ML for capturing sequential patterns in patient timelines; LSTMs process the raw event sequence (HbA1c readings, encounters, medications over time) rather than engineered features; requires padding and masking for variable-length patient histories; evaluated against XGBoost on the same patient-level split
- Survival models explored: Kaplan-Meier for time-to-event visualization; Cox Proportional Hazards for feature-weighted survival analysis — informative for understanding time-to-adverse-event distributions but not used in production pipeline due to complexity of real-time scoring
- Final model selection: XGBoost chosen for highest PR-AUC, native NaN handling (critical given missing lab values in real EHR data), production-grade inference speed, and compatibility with SHAP TreeExplainer; LSTM performance was competitive but required substantially more data preprocessing and lacked native explainability

### 2.4.3 Handling Class Imbalance
- Class imbalance: approximately 3% of patient-day instances are positive (high-risk event within 30 days)
- Approaches evaluated: class_weight='balanced' (logistic regression), scale_pos_weight (XGBoost), SMOTE (synthetic oversampling — explored but found to create unrealistic synthetic patient timelines), threshold adjustment post-training
- Selected approach: scale_pos_weight in XGBoost combined with threshold tuning; avoids synthetic data artifacts while directly addressing the optimization objective

## 2.5 Clinical Text Processing Architecture

### 2.5.1 Note Cleaning Pipeline
- HTML artifact stripping: clinical notes imported from web-based EHR systems often contain residual HTML tags (e.g., <br>, &amp;); strip_html() removes these before extraction
- OCR error correction: scanned documents introduce systematic character substitutions (e.g., 'l' → '1', 'O' → '0'); fix_ocr_text() applies a configurable substitution dictionary; has_ocr_errors() detects affected records; valid_mixed_terms list prevents correction of legitimate alphanumeric medical codes (e.g., "HbA1c", "T2DM")

### 2.5.2 Regex-Based Extraction
- Vitals extracted: systolic and diastolic blood pressure (LOINC 8480-6, 8462-4), body weight in kg/lbs, height in cm, SpO2, temperature in Celsius and Fahrenheit
- Labs extracted: HbA1c (LOINC 4548-4, range-validated 4.0–15.0%), glucose (LOINC 2339-0), creatinine (LOINC 2160-0), eGFR (LOINC 33914-3)
- Medications extracted: 19 known diabetes and comorbidity medications (Metformin, Insulin variants, SGLT2 inhibitors, GLP-1 agonists, antihypertensives)
- Range validation: extracted numeric values are checked against clinical reference ranges; out-of-range values are discarded as extraction errors rather than propagated as data quality issues

### 2.5.3 NER with spaCy and scispaCy (Extension)
- en_core_web_sm: general English NER for document structure and date extraction
- en_ner_bc5cdr_md: biomedical NER model (diseases and chemicals); identifies disease mentions and drug names beyond the regex dictionary
- Role in pipeline: NER supplements regex extraction for entity types not covered by fixed patterns; particularly useful for synonymous medication mentions and uncommon condition descriptions

## 2.6 Patient Profile and Data Model Architecture
- PatientProfile class: consolidates all structured and extracted data per patient; stores demographics, conditions, medications, observations, and encounter timelines; identifies diabetes type from condition descriptions using normalized text matching against DIABETES_TERMS vocabulary
- ExtendedPatientProfile: extends PatientProfile with time-series timelines (HbA1c, BP, eGFR, BMI, medication events, emergency visits, hospitalizations, care gaps, complications); implements get_daily_features(cutoff_time) for temporally-safe feature computation; implements generate_all_instances() for building the training dataset
- FHIR-inspired structure: patient profile schema mirrors HL7 FHIR resource types (Patient, Observation, Condition, MedicationRequest); LOINC codes attached to all extracted observations; ICD-10 codes preserved from structured tables; simplified FHIR representation rather than full FHIR JSON serialization
- Risk labeling: high-risk event definition based on ADA 2025 Standards — ED visit (SNOMED 50849002), inpatient admission (SNOMED 32485007, 183452005, 305351004), acute complications (ketoacidosis, hypoglycemia, MI, stroke, amputation), or HbA1c ≥ 10.0% within the subsequent 30-day window

## 2.7 Monitoring System Architecture
- EHRPipeline class: orchestrates all stages (load CSVs → clean notes → extract notes → build profiles → predict / explain); designed for both batch evaluation and single-patient real-time query
- predict_patient(patient_id, date): computes features at the given date and returns a probability score; O(1) once profiles are loaded
- explain_patient(patient_id, date): adds SHAP-based decomposition of risk and protective factors with clinical language translations; returns structured JSON with risk_level (LOW / MODERATE / HIGH / CRITICAL), top contributing features, and per-feature clinical meanings
- Autonomous monitoring workflow: continuously re-scores patient risk as new EHR data arrives; no manual trigger required; designed to run on batch EHR update cadence (e.g., nightly or per-encounter)
- Automated alert generation: configurable risk thresholds (CRITICAL ≥ 80%, HIGH ≥ 60%, MODERATE ≥ 40%, LOW < 40%); allows clinical teams to tune precision-recall trade-off per ward or patient population
- Simulation environment: test harness built to replay historical patient data through the pipeline under controlled conditions; used to validate alert timing, threshold sensitivity, and system behavior under edge cases (patients with no recent labs, patients with very long care gaps, high-volume simultaneous alerts)
- Streamlit monitoring dashboard: pipeline overview (patient counts, observation counts), individual patient timeline explorer, model performance visualization, feature importance view

## 2.8 System Architecture and Business Constraints
- End-to-end data flow: raw EHR CSV + clinical notes → OCR/HTML cleaning → regex + NER extraction → FHIR-structured patient profiles → 22-feature computation → XGBoost risk scoring → SHAP explanation → alert dashboard
- API design: EHRPipeline exposes predict_patient(patient_id, date) and explain_patient(patient_id, date) as the core query interface; batch scoring via build_feature_matrix(); all interfaces return structured Python dicts / DataFrames suitable for downstream API wrapping
- Computational requirements: profile build is a one-time O(N patients) operation; single-patient prediction is O(1) post-build; SHAP explainer initialization is lazy (first call only); designed to run on standard hospital server hardware without GPU requirement
- Business constraints the system must meet: alerts must be actionable and understandable to clinicians making urgent decisions (SwissCare Clinical Safety Policy); clinicians must see rationale before acting (SwissCare Clinical Accountability Policy); system must operate within EU/Switzerland data protection perimeter; all outputs must be auditable for regulatory inspection

---

# 3. Testing and Validation Protocol

## 3.1 Model Performance Results

### 3.1.1 Baseline Progression
- Random prediction: ROC-AUC ~0.50, PR-AUC ~0.03 (class distribution)
- Majority class: accuracy ~97%, ROC-AUC ~0.50, PR-AUC negligible — demonstrates the deceptiveness of accuracy as a metric
- HbA1c > 9% threshold: PR-AUC ~0.155 — establishes the floor that a learned model must beat to add value over clinical rules alone
- Risk factor counting: modest improvement over single threshold; confirms multi-feature combination is meaningful
- Logistic Regression (6 features): establishes linear upper bound for the feature set

### 3.1.2 Advanced Model Comparison (22 features)
- Decision Tree: interpretable but overfits; lower generalization performance
- Random Forest: improved over Decision Tree via ensemble; reduces variance
- XGBoost (tuned): highest ROC-AUC and PR-AUC; selected as production model
- Improvement from 6-feature baselines to 22-feature XGBoost: quantify delta in PR-AUC to justify feature engineering effort

### 3.1.3 Clinical Alert Performance Analysis
- For every 100 alerts generated at the chosen operating threshold: X readmission-relevant events captured, Y false alarms
- Precision at 80% recall: reported as primary threshold-specific metric
- Comparison with HbA1c-only rule: how many additional at-risk patients does the ML system identify that the simple threshold would have missed?

## 3.2 Generalization and Fairness Assessment

### 3.2.1 Subgroup Performance Analysis
- Performance stratified by age group (pediatric, adult, elderly)
- Performance stratified by gender
- Performance stratified by race/ethnicity
- Metric reported per subgroup: ROC-AUC, PR-AUC, True Positive Rate (Sensitivity), False Positive Rate

### 3.2.2 Fairness Metrics
- Equal Opportunity: equal sensitivity (TPR) across demographic groups — does the model catch high-risk events at the same rate regardless of age, gender, or ethnicity?
- Predictive Parity: equal precision (PPV) across groups — is a high-risk alert equally reliable regardless of patient demographics?
- Equalized Odds: combined TPR and FPR parity
- Disparity Analysis: identify which group shows the largest performance gap and characterize the direction (over-prediction vs. under-prediction)

### 3.2.3 Calibration Analysis
- A well-ranked model (high ROC-AUC) can still be poorly calibrated: a predicted 80% risk should correspond to an 80% observed event rate
- Calibration curve plotted: predicted probability vs. observed frequency across probability bins
- Clinical significance: miscalibration affects clinical decision-making — clinicians acting on "80% risk" alerts need that number to be meaningful; poor calibration erodes clinician trust
- Post-hoc calibration options: Platt scaling, isotonic regression — note whether applied and impact

## 3.3 Robustness Testing

### 3.3.1 Missing Data Robustness
- EHR data routinely has missing observations: patients without recent HbA1c (gaps in monitoring), no eGFR if kidney function was not tested, no BP from notes if OCR extraction failed
- XGBoost's native NaN handling: model was trained and evaluated with missing values propagated; no imputation applied to avoid introducing systematic bias
- Robustness test: artificially increase missing data rates per feature and observe PR-AUC degradation curve; identifies which features are most critical and where imputation may be warranted in deployment

### 3.3.2 Temporal Validation
- Standard GroupShuffleSplit (random patient split) may overestimate real-world performance because training and test patients are drawn from the same time period
- Temporal split: patients assigned to train or test based on their first appearance date — training on historical patients, testing on patients who appear later; simulates real deployment where model is trained on past data and applied to future patients
- Expected finding: temporal performance typically lower than random split; quantify the gap; assess whether the model degrades gracefully or shows cliff-edge failure
- Temporal leakage check: all feature timestamps verified to precede prediction date; no future lab values or encounter data can appear in features

### 3.3.3 Cross-Validation Stability
- GroupKFold cross-validation: 5-fold with strict patient-level group separation
- Reports mean and standard deviation of ROC-AUC and PR-AUC across folds
- High variance across folds would indicate overfitting or sensitivity to the specific patient population in each fold

## 3.4 Clinical Validity Assessment

*This section is required by the project briefing as a named deliverable.*

### 3.4.1 Risk Prediction Calibration
- For every 100 alerts generated at the chosen operating threshold: X readmission-relevant events captured, Y false alarms — provide concrete figures from model evaluation
- Frame the trade-off in clinical terms: a 20% precision means 4 out of 5 nurses called to act will find the patient is not in acute danger; a 40% precision halves that burden
- Compare alert yield against the HbA1c-only rule: how many additional at-risk patients does the ML system surface that the simple threshold would have missed?
- Assess whether the 48–72 hour lead-time objective is met: what is the median time between the first CRITICAL alert and the subsequent adverse event in the evaluation set?

### 3.4.2 Demographic Fairness
- Model performance reported across age groups (e.g., under 40, 40–65, over 65), gender, and ethnicity subgroups
- Key question: are certain patient groups systematically under-alerted (low recall) or over-alerted (low precision)?
- Report the largest observed performance gap and its direction; discuss whether it reflects a genuine clinical difference in risk patterns or a model bias to be mitigated
- Note limitations of fairness assessment on synthetic Synthea data: demographic distributions may not reflect the real SwissCare patient population

## 3.5 Error Analysis and Model Limitations
- False negative analysis: what do the missed high-risk patients look like? Common patterns — patients with sparse EHR data (low feature completeness), unusual comorbidity profiles, or very recent first encounters
- False positive analysis: what drives spurious high-risk alerts? Common patterns — patients with high medication counts but stable HbA1c, elderly patients with many care encounters but no acute deterioration
- Feature completeness impact: performance as a function of missing feature rate; patients with fewer than X non-null features should trigger a "data insufficient for confident scoring" warning rather than a low-confidence score
- Known model limitations: trained on synthetic data; no real-world validation; performance on Type 1 diabetes patients is limited by their lower representation; system is not validated for pediatric patients or gestational diabetes

---

# 4. Regulatory Compliance Assessment

## 4.1 EU AI Act (High-Risk Medical AI Classification)
- Classification rationale: the system is a safety component of a clinical decision support tool; it generates patient risk alerts that directly influence clinician intervention decisions; this falls squarely within the EU AI Act Annex III high-risk AI categories for health
- Intended purpose documentation: system is designed to predict 30-day adverse events in diabetic patients using historical EHR data; it is a decision-support tool, not an autonomous decision-maker; final clinical decisions remain with treating clinicians
- Known limitations documented: model trained on synthetic Synthea data (distribution may differ from real patient populations); performance on rare diabetes subtypes (Type 1, gestational) is limited by training data; system does not process real-time physiological monitoring (only EHR batch updates)
- Data quality measures: input validation on all extracted lab values (range checks for HbA1c, glucose, creatinine, eGFR); OCR error detection and correction pipeline; HTML artifact stripping; duplicate record deduplication
- Accuracy and robustness testing: AUROC, PR-AUC, calibration, subgroup fairness, temporal validation — all documented in Section 3; results retained as evidence
- Human oversight integration: all predictions are presented as probability scores with SHAP explanations, never as autonomous clinical orders; CRITICAL and HIGH alerts surface to clinicians for review; alert thresholds are configurable by clinical teams; clinicians can override or dismiss any alert with logged rationale
- Event logging: EHRPipeline records prediction date, patient ID, risk score, and contributing features per alert; enables post-market monitoring of real-world performance vs. development-time performance
- Post-market monitoring design: periodic re-evaluation of alert precision/recall against actual patient outcomes; monitoring for distribution shift between training population and live patient population

## 4.2 GDPR and revFADP (Health Data Protection)
- Automated decision classification: the system generates risk alerts that could influence care eligibility and clinical prioritization decisions; this falls within the scope of GDPR Article 22 / revFADP automated decision provisions
- Meaningful information obligation: for every alert generated, the explain_patient() function produces a structured explanation including top risk-increasing factors, top protective factors, the probability score, and natural-language clinical meanings for each feature; this explanation is surfaced to the clinician prior to any action being taken
- Human review process: all alerts are reviewed by a qualified clinician before any care intervention is triggered; the monitoring dashboard does not allow the system to autonomously schedule interventions; nursing staff see alert rationale before acting
- Right to contest: patients are informed (via clinical transparency policies) that their risk score is computed from their EHR data; clinicians can flag alerts as incorrect through the dashboard, triggering a case review
- Data minimization: only the 22 clinically necessary features are passed to the model; raw unstructured notes are not stored after extraction; patient identifiers are pseudonymized (UUID) throughout the pipeline
- Cross-border data handling: data processed within the EU/Switzerland data protection perimeter; no patient data transmitted to external services during model inference (inference runs locally on hospital infrastructure)

## 4.3 HL7 FHIR Compliance
- All patient profiles follow FHIR-inspired resource schemas: Patient (demographics), Observation (labs and vitals with LOINC codes), Condition (diagnoses with ICD-10 codes), MedicationRequest (prescriptions)
- LOINC codes assigned to all extracted observations: HbA1c (4548-4), glucose (2339-0), creatinine (2160-0), eGFR (33914-3), systolic BP (8480-6), diastolic BP (8462-4), body weight (29463-7), body temperature (8310-5), SpO2 (2708-6)
- Simplified FHIR representation used (as specified in project brief); full FHIR JSON serialization is a natural extension for production deployment and cross-border exchange
- Interoperability benefit: standardized resource format means extracted data from clinical notes integrates seamlessly with structured table data, and output profiles could be consumed by other FHIR-compliant systems within the SwissCare network

## 4.4 SNOMED CT and ICD-10 Standards
- ICD-10 codes preserved from structured Synthea tables for all patient conditions; no recoding performed, ensuring compatibility with hospital billing and cross-institutional records
- SNOMED CT codes used for high-risk event definitions: ED visit (50849002), inpatient admissions (32485007, 183452005, 305351004) — these definitions are consistent with published clinical quality measure specifications
- Extracted NER entities (disease mentions, medication mentions) are mapped back to SNOMED/ICD vocabulary where possible; un-mappable free-text terms are logged rather than silently dropped
- Clinical coding accuracy rationale: standardized terminology allows alert output to be interpreted by any clinician regardless of which hospital system they trained in

## 4.5 SwissCare Clinical Policies
- Clinical Safety Policy ("actionable and understandable to clinicians making urgent decisions"): met through the SHAP explanation layer — every alert includes the top 5 risk-increasing factors with plain-language descriptions and the patient's specific feature values; the explain_patient() output is designed to be readable in under 30 seconds
- Clinical Accountability Policy ("clinicians must know the rationale before acting"): met through mandatory display of the explanation report in the monitoring dashboard before any action button is accessible; the dashboard UI does not allow dismissal without viewing the explanation

---

# 5. Lessons Learned and Failed Approaches

## 5.1 Failed Experiment: SMOTE for Class Imbalance

**Hypothesis:** Synthetic Minority Oversampling Technique (SMOTE) would improve model performance by generating synthetic high-risk patient instances and balancing the training dataset.

**Methodology:** SMOTE applied to the 22-feature training matrix after patient-level splitting; evaluated against XGBoost with scale_pos_weight on the same test set.

**Results:** [quantitative PR-AUC comparison]

**Analysis of failure:** SMOTE operates on feature vectors in isolation, generating synthetic "patients" by interpolating between real patient instances. For time-series healthcare features (e.g., days_since_last_encounter, hba1c_trend_last_180d), this produces combinations that are clinically implausible — for example, a synthetic patient with a large positive HbA1c trend but a very recent last encounter, which rarely co-occurs in real clinical trajectories. The synthetic instances distort the feature space and the model learns patterns that don't generalize.

**Takeaway:** In healthcare ML, class imbalance should be addressed through loss-function weighting (scale_pos_weight) or threshold adjustment rather than data augmentation; temporal feature spaces violate the IID assumptions that SMOTE relies on.

## 5.2 Failed Experiment: Row-Level Train/Test Split

**Hypothesis:** Standard random row-level splitting would produce a valid evaluation of model generalization.

**Methodology:** scikit-learn train_test_split applied to the feature matrix without grouping by patient_id.

**Results:** ROC-AUC and PR-AUC appeared substantially higher than patient-level split results.

**Analysis of failure:** The same patient has multiple rows in the dataset (one per day observed). With row-level splitting, Patient A's Monday observation could be in training and Patient A's Tuesday observation in the test set. The model learns patient-specific patterns (e.g., that a particular HbA1c trajectory belongs to a specific patient) and "recognizes" that patient in the test set. This is data leakage masquerading as good performance. In deployment, the model would encounter entirely new patients it has never seen — the artificially inflated metrics would not hold.

**Takeaway:** Always split at the patient level (GroupShuffleSplit with patient_id as the group key) before any preprocessing; imputation fitted on train must be applied to test separately to avoid further leakage.

## 5.3 Failed Experiment: Temporal Features Without Cutoff Enforcement

**Hypothesis:** Computing features using all available patient data (without cutoff_time) would simplify implementation and not meaningfully affect results.

**Methodology:** A version of get_daily_features() was initially implemented without the cutoff_time parameter, using the full timeline for each feature.

**Results:** Training performance appeared excellent; temporal validation performance collapsed.

**Analysis of failure:** Without cutoff enforcement, features for a "prediction" on 2017-03-15 could include HbA1c readings from 2017-06-01 (three months in the future from the perspective of a clinician at that date). The model learned to predict adverse events using information that would only be available after the event — perfectly circular reasoning. The temporal validation exposed this because it requires predicting on patients entirely absent from training, making future-data features impossible to exploit.

**Takeaway:** Temporal safety is the single most critical correctness requirement in time-series healthcare ML. The cutoff_time parameter must be enforced at every feature computation, not as an afterthought.

## 5.4 Failed Experiment: ZIP-Code Census Features

**Hypothesis:** Adding socioeconomic and healthcare access context at the ZIP-code level would improve 30-day adverse event prediction by capturing the environmental factors that influence patient health outcomes — population density, healthcare provider density, and insurance coverage.

**Methodology:** Three ZIP-code-level features were constructed from US Census ACS data and CMS NPPES provider data and merged into the feature matrix by patient ZIP code: density_per_sq_mile (population per square mile from ACS 5-year estimates), overall_healthcare_coverage (diabetes-relevant provider density, weighted by specialty taxonomy from NPPES), coverage_score (composite insurance coverage score: employer, direct-purchase, Medicare, Medicaid coverage percentages). The 25-feature model (22 clinical + 3 census) was evaluated against the 22-feature clinical-only model.

**Results:** Pearson correlations between the three census features and the 30-day high-risk event label were essentially zero: density_per_sq_mile r = −0.0058, overall_healthcare_coverage r = +0.0050, coverage_score r = +0.0034. XGBoost and logistic regression both assigned near-zero weight to the census features. PR-AUC did not improve materially over the 22-feature model. Note: coverage_score had 45.8% missing values, filled with 0 via fillna(0), which introduced an artificial cluster at zero that muddled the signal further.

**Analysis:** The 30-day prediction window is driven almost entirely by the patient's current clinical state — their HbA1c trend, eGFR decline, recent ED visits. Whether a patient lives in a dense urban ZIP or a rural one does not meaningfully change what will happen to them in the next 30 days once strong clinical features are already present. Logistic regression's L2 regularization correctly shrinks near-zero coefficients to zero. XGBoost tree splits require some signal to act on — with correlations this close to zero, there is no threshold to split on. Census features would be more informative for longer-horizon outcomes (1-year hospitalization, access to follow-up care) where systemic access barriers compound over time.

**Takeaway:** "ZIP-code-level census features (population density, provider density, insurance coverage) add no predictive power to a 30-day high-risk event model when strong clinical features are already present." This is a valid and interpretable scientific finding, not a failure of implementation. The 22-feature clinical-only model is the correct model to deploy; the 25-feature pipeline remains in the codebase for completeness and reproducibility.

## 5.5 Lessons Learned

### On Data Quality
- EHR data quality issues are not edge cases — duplicates, OCR errors, HTML artifacts, and orphaned records are the norm, not the exception; pipeline robustness must be designed before modeling begins, not after
- Range validation on extracted clinical values (HbA1c < 4.0% or > 15.0%) catches extraction errors early; without it, nonsensical values propagate silently through the feature matrix

### On Modeling
- The gap between clinical domain knowledge and ML feature engineering is smaller than expected; clinically meaningful features (care gap counts, HbA1c trends) outperformed purely statistical feature selection approaches
- XGBoost's native NaN handling was a meaningful practical advantage over alternatives that required imputation strategies for missing labs
- SHAP global feature importance is useful not just for reporting but as a debugging tool — unexpectedly high-importance features can reveal data artifacts (e.g., encounter records with future dates)

### On Evaluation
- PR-AUC is a more honest metric than ROC-AUC for rare event prediction — it should be the headline metric whenever positive class prevalence is below 10%
- Calibration is easy to neglect but clinically essential; a model that outputs "80% risk" when the true rate is 40% will erode clinician trust quickly once deployed

### On Interpretability
- Translating SHAP feature names into clinical language (CLINICAL_MEANINGS dictionary) was a small engineering investment with a large impact on dashboard usability
- Per-patient SHAP explanations (risk factors + protective factors) address the GDPR meaningful information requirement and the SwissCare Clinical Accountability Policy simultaneously — interpretability is both a regulatory requirement and a clinical usability feature

---

# 6. Outlook

## 6.1 Recommendations for Similar Projects
- Invest in temporal safety infrastructure (cutoff_time enforcement) from day one — it is far harder to retrofit than to build correctly upfront
- Design the evaluation framework (metrics, splitting strategy, fairness checks) before writing the first model; the evaluation choices are as consequential as the model choices
- For high-risk medical AI: treat regulatory documentation as a co-equal deliverable alongside the model, not as a post-hoc addition; EU AI Act and GDPR requirements shape architecture decisions (human oversight, logging, explainability)
- External data sources (census, socioeconomic) add value for long-horizon predictions but should be validated for signal before integration; near-zero correlations at the instance level indicate environmental features are absorbed by clinical features at 30-day horizons
- **Calibrate the baseline before opening extensions:** if the instructor-provided implementation already achieves ~80% of achievable performance, extensions should be explicitly scoped to add dimensions of value (interpretability, fairness, deployment robustness) rather than marginal metric gains; students should know upfront that the delta may be small and that this is expected, not a failure

## 6.2 Gap Between Course Expectations and Reality

### What worked well
- The pipeline architecture provided by the course was genuinely well-structured: clean separation between data processing, profile construction, and modeling stages made it straightforward to extend incrementally across weeks
- The week-by-week homework design built real understanding of why each component (temporal safety, patient-level splitting, PR-AUC) exists — the order of concepts was pedagogically sound

### What was harder than expected
- **Development cycle time:** the `pip install -e src/` step required after every change to source files in Colab took approximately 5–15 minutes per iteration, making tight feedback loops on code changes difficult; this is a structural friction point in the Colab + Google Drive setup that meaningfully slowed experimentation
- **Diminishing returns on the core model:** the XGBoost implementation proposed in the course materials was already very well-tuned, leaving limited room for meaningful performance improvement; by the time extensions were explored, the best achievable PR-AUC improvement was marginal, which made it hard to maintain motivation for further model optimization
- **Implication for course design:** extensions that offer meaningful incremental gains (interpretability dashboards, fairness auditing, simulation testing, deployment hardening) are more motivating than pure metric-chasing once the baseline is already strong; a clearer framing that "the baseline is intentionally ~80% of the ceiling — your job is to add value in other dimensions" would set better expectations from the start

### On the Gap Between Expectations and Reality
- [To be completed by team: any other surprises, e.g. on regulatory complexity, clinical domain knowledge needed, or collaborative aspects of the project]
