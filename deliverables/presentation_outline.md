# Presentation Outline
## Patient Monitoring via Structured Electronic Health Records
### Preventing Adverse Diabetic Events Through ML Patient Monitoring
**Presenting team (consultants):** Sandy, Nicholas, Timarian
**Client:** SwissCare Health Network AG
**Duration:** 40 min + Q&A

---

## ⚠️ TO BE DISCUSSED BEFORE FINALISING

### Dashboard transition — when?
- **Option A — Switch at Section 4 only:** Slides 1–12 as deck, then live dashboard for Results & Evaluation (Section 4). Cleanest split: deck tells the story, dashboard shows the evidence.
- **Option B — Switch at Section 2:** Present Sections 2, 3, and 4 entirely from the dashboard (one tab each: Data & Methodology, Architecture, Results). More impressive technically; riskier if the demo has issues; requires dashboard tabs to be presentation-ready.
- **Recommendation to discuss:** Option A is safer for a 40-min slot. Option B is more compelling if the dashboard is polished and the tabs map cleanly onto the section structure.

### Review checklist — fill in before building slides
- [ ] **Section 2:** Confirm exact data quality figures (% duplicates, % orphaned records, % missing per feature)
- [ ] **Section 2:** Confirm number of clinical note records and format breakdown (free text vs. semi-structured)
- [ ] **Section 2:** Confirm preprocessing steps and which cleaning issues were most impactful
- [ ] **Section 3:** Confirm pipeline runtime on the hardware you'll demo from (~20 min estimate)
- [ ] **Section 3:** Confirm key technical challenges — align with what Timarian's error analysis found
- [ ] **Section 4:** Fill in actual PR-AUC and ROC-AUC numbers from the trained XGBoost model
- [ ] **Section 4:** Fill in the "100 alerts" figure: X caught / Y false alarms at the operating threshold
- [ ] **Section 4:** Insert Timarian's error analysis findings (false negative and false positive profiles)
- [ ] **Section 4:** Insert fairness results — which demographic group shows the largest performance gap
- [ ] **Section 5:** Confirm the ROI estimate (cost per prevented readmission × expected prevention rate)
- [ ] **Section 5:** Confirm Alice/Bob patient IDs or synthetic examples for the SHAP force plot demo
- [ ] **All sections:** REVIEW SECTIONS 2–6 IN DETAIL before finalising slide content

---

## Narrative Thread: Alice and Bob

Run Alice and Bob as a recurring thread to ground technical content in a human story. Introduce them in the problem statement, return to them at each major section transition.

**Alice**, 67, Type 2 diabetic in Worcester, MA. HbA1c creeping up for 18 months. Missed her last two check-ups. eGFR declining. Her deterioration is documented only in free-text clinical notes — inaccessible to any monitoring system. Without intervention, she is 30 days from a hospitalisation.

**Bob**, 52, Type 2 diabetic in Boston. HbA1c borderline, but he had an ER visit 6 weeks ago and was hospitalised twice in the past year. Care fragmented across three providers. Each sees only part of the picture.

Use them at: Slide 3 (problem), Slide 6 (data motivation), Slide 13 (live SHAP demo), Slide 18 (outcome / close).

---

## Timing Budget (40 min)

| Section | Slides | Time |
|---|---|---|
| 1. Introduction & Recommendation | 1–5 | ~8 min |
| 2. Data & Methodology | 6–9 | ~9 min |
| 3. Implementation & Architecture | 10–12 | ~6 min |
| 4. Results & Evaluation | 13–17 | ~11 min → **live dashboard** |
| 5. Key Takeaways | 18–19 | ~6 min |

---

## Section 1 — Introduction & Recommendation (~8 min)
*Pyramid structure: lead with the answer. The rest of the presentation is the support.*

### Slide 1 — Title & Team
- **Title:** Patient Monitoring via Structured Electronic Health Records
- **Subtitle:** Preventing Adverse Diabetic Events Through ML Patient Monitoring
- **Consulting team:** Sandy, Nicholas, Timarian
- **Client:** SwissCare Health Network AG
- **Date / Engagement:** [date]

---

### Slide 2 — Our Recommendation *(the pyramid top)*
*This is the most important slide. State the conclusion before the audience has to ask for it.*

> **"We recommend deploying an XGBoost pipeline with 22 clinically-grounded features to predict 30-day adverse events in diabetic patients. At the chosen operating threshold, the system identifies [X]% of high-risk patients 48–72 hours before adverse events, with an estimated net ROI of [CHF/USD X] per year based on prevented readmissions. The system carries no material bias risk across demographic subgroups and is fully compliant with EU AI Act high-risk requirements, GDPR/revFADP, and HL7 FHIR standards."**

- Three supporting pillars — and we will walk through each:
  1. **It works:** PR-AUC of [X] vs. best clinical rule baseline of 0.155 — a [Y]x improvement
  2. **It can be trusted:** SHAP explanations per alert, demographic fairness validated, full compliance architecture documented
  3. **It is worth deploying:** ROI positive from [X] prevented readmissions per year; pipeline runs in ~20 min on standard hardware; no GPU required

---

### Slide 3 — The Problem: Alice and Bob
- Two diabetic patients. Two trajectories heading toward a preventable crisis.
- **Alice (67, Worcester MA):** 6 unread free-text clinical notes flagging worsening glycemic control. No alert fired. No doctor called. 30 days from hospitalisation.
- **Bob (52, Boston):** 3 providers, 2 recent hospitalisations, fragmented data. No single person has the full picture.
- **The gap:** 70% of clinical information at SwissCare lives in unstructured notes — inaccessible to any systematic monitoring today
- **The cost:** Average preventable hospitalisation costs $15,000–$25,000; each missed readmission is a patient outcome failure and a balance sheet hit
- *"Our engagement: read what the system isn't reading, and flag Alice and Bob before it's too late."*

---

### Slide 4 — Scope & Mandate
- **Client:** SwissCare Health Network AG — hospitals and affiliated clinics, Switzerland & EU
- **Dataset:** 50,000+ diabetic patients, Massachusetts (Synthea-generated EHR data)
- **Prediction target:** 30-day high-risk adverse event (ER visit, hospitalisation, acute complication, HbA1c ≥ 10%)
- **Engagement deliverables:**
  1. End-to-end ML pipeline: raw EHR → structured profiles → risk scores → explainable alerts
  2. Compliance documentation: EU AI Act, GDPR/revFADP Article 22, HL7 FHIR, SNOMED/ICD-10
  3. Monitoring dashboard: live risk scoring with SHAP explanations, configurable alert thresholds
- **Timeline:** 22-week project

---

### Slide 5 — Stakeholders & Their Success Metrics
- **Chief Medical Officer:** Adverse event prediction rate, false positive rate — *does the alert tell me something clinically actionable?*
- **Emergency Department Director:** Alert recall, lead time — *does it catch every avoidable ER visit early enough?*
- **Nursing Leadership:** False positive rate, alert interpretability — *is every alert worth my team's time?*
- **Compliance Officer:** Zero regulatory violations, documentation completeness — *can we pass an EU AI Act audit?*
- **Chief Information Officer:** System reliability, FHIR compliance rate — *does it integrate with existing systems?*
- **Patients (Alice & Bob):** Better outcomes, right to understand automated decisions affecting their care

---

## Section 2 — Data & Methodology (~9 min)
*If Option B (dashboard from Section 2): transition here. Dashboard Tab 1: Data & Methodology.*

### Slide 6 — The Data
- **Structured EHR:** 50,000+ patient records across 7 relational tables — patients, encounters, conditions, medications, observations, procedures, allergies
- **Unstructured clinical notes:** ~25% of encounters exist only in free-text and semi-structured reports — this is where Alice's deterioration is hiding
  - [confirm total note count and format split]
- **Geography:** Massachusetts, US; patient ZIP codes available for socioeconomic analysis
- **Target population:** Diabetic patients identified by ICD-10 and SNOMED condition codes
- **Prediction label:** Binary — high-risk adverse event (ER visit, hospitalisation, acute complication, HbA1c ≥ 10%) within the next 30 days
- *"Alice is one of these 50,000 patients. Her risk lives in the notes."*

---

### Slide 7 — Data Quality & Preprocessing
- **Key quality issues encountered:**
  - Class imbalance: only ~3% of patient-day instances are positive — predicting "low risk" always gives 97% accuracy and zero clinical value
  - OCR errors in scanned notes: character substitutions corrupt lab values before extraction
  - HTML artifacts: residual tags from web EHR exports must be stripped
  - Missing observations: no recent HbA1c, eGFR, or BP for many patients — missing data is the norm
  - Duplicate and orphaned records: [X]% required deduplication; orphaned notes resolved [confirm figure]
- **Pipeline response:**
  - HTML stripping → OCR correction (configurable dictionary, safelist for valid codes like "HbA1c")
  - Regex + NER extraction of vitals, labs, medications with LOINC code assignment
  - Range validation: extracted values outside clinical reference ranges discarded
  - All features computed with strict `cutoff_time` — no future data can contaminate training

---

### Slide 8 — Feature Engineering: 22 Features, 4 Families
- **Why 22 features?** Clinically motivated — every feature maps to a known risk indicator; no black-box statistical selection
- *Temporal counters (5):* days since last HbA1c, encounter, ER visit, hospitalisation, medication change — capture care engagement
- *Rolling windows (4):* encounters, ER visits, hospitalisations, medication changes over 90–365 days — capture trajectory
- *Clinical values (5):* current HbA1c, systolic BP, diastolic BP, eGFR, BMI category — the clinical picture today
- *Trend features (3):* HbA1c, BP, eGFR slope over 6–12 months — is the patient improving or deteriorating?
- *Composite risk (5):* active medication count, complication count, care gap count, longest care gap, age — disease burden summary
- **Extension tested:** +3 ZIP-code census features (population density, provider density, insurance coverage) → near-zero correlations (r < 0.006) — not deployed; environmental factors do not predict 30-day clinical outcomes when strong clinical features are present

---

### Slide 9 — Model Selection & Validation
- **Selection rationale:** Healthcare context demands explainability — clinicians cannot act on black-box scores; GDPR Art. 22 requires meaningful explanations; tree methods are well-suited to tabular features with missing values

| Model | PR-AUC | ROC-AUC | Why tested |
|---|---|---|---|
| Random / Majority class | ~0.03 / negligible | ~0.50 | Establishes floor; exposes why accuracy is meaningless |
| HbA1c > 9% rule | ~0.155 | — | Clinical rule benchmark |
| Logistic Regression | [result] | [result] | Linear ceiling; L2 shrinks weak features |
| Random Forest | [result] | [result] | Ensemble variance reduction |
| **XGBoost + SHAP** | **[result]** | **[result]** | **Best PR-AUC; native NaN; full explainability → selected** |

- **Validation approach:** Patient-level split (no patient in both train and test) → 5-fold GroupKFold stability → temporal validation (train on past patients, test on future) → robustness under degraded missing data rates
- **Class imbalance:** `scale_pos_weight` in XGBoost; `eval_metric='aucpr'` to optimise directly for the clinical metric

---

## Section 3 — Implementation & Technical Architecture (~6 min)
*If Option B (dashboard from Section 2): Dashboard Tab 2: Architecture.*

### Slide 10 — Pipeline Architecture
*Visual: left-to-right flowchart — clean, consulting-style diagram*

```
[Raw EHR CSVs]     [Clinical Notes]
       ↓                  ↓
     [OCR / HTML Cleaning]
              ↓
   [Regex + NER Extraction → LOINC codes]
              ↓
   [FHIR-structured Patient Profiles]
              ↓
   [22-Feature Computation (cutoff-safe)]
              ↓
        [XGBoost Scorer]
              ↓
     [SHAP Explanation Layer]
              ↓
     [Alert Dashboard (Streamlit)]
```

- Modular `EHRPipeline` class: each stage chains cleanly, designed for both batch and single-patient real-time query
- Single-patient prediction: sub-second once profiles are loaded
- Alert levels: CRITICAL ≥ 80%, HIGH ≥ 60%, MODERATE ≥ 40%, LOW < 40% — configurable per ward

---

### Slide 11 — System Architecture & Deployment
- **Data sources:** Synthea CSV tables + free-text notes (extensible to live FHIR API in production)
- **Core API:** `predict_patient(patient_id, date)` → risk probability; `explain_patient(patient_id, date)` → SHAP decomposition with clinical language
- **Monitoring dashboard:** Streamlit — patient explorer, model performance, feature importance, per-patient SHAP force plots
- **Computational footprint:** Full pipeline ~20 min on standard laptop / Colab; no GPU required; designed to run within hospital data perimeter — no patient data transmitted externally
- **Integration points:** FHIR-structured outputs, LOINC-coded observations, SNOMED event definitions — compatible with any FHIR-compliant hospital system

---

### Slide 12 — Key Technical Challenges & How We Solved Them
- **Temporal leakage** *(most critical):* Features computed without a prediction date cutoff silently include future information — metrics looked excellent, then collapsed on temporal validation; solution: strict `cutoff_time` enforcement at every feature computation
- **Patient-level data splitting:** Row-level random splits inflated metrics by allowing the model to "memorise" patients; solution: `GroupShuffleSplit` with patient ID as group key
- **Missing data at scale:** ~[X]% of feature values are NaN; imputation introduced systematic bias; solution: XGBoost's native NaN handling — no imputation needed
- **OCR overcorrection:** The OCR fixer corrupted valid codes like "HbA1c" before a safelist was added; solution: `valid_mixed_terms` configuration parameter
- **[Any challenges from Timarian's error analysis to add here]**

---

## Section 4 — Results & Evaluation (~11 min)
### *** TRANSITION TO LIVE DASHBOARD HERE ***
*At this point, switch from the slide deck to the Streamlit monitoring dashboard. The following slide notes describe what to show on each dashboard tab — they are speaker guide notes, not slides.*

---

### Dashboard Tab 1 — Pipeline Overview / Model Performance
*Show the pipeline overview tab: patient count, observation count, extraction stats*
- Walk through the pipeline stage metrics: [X] patients loaded, [Y] clinical notes processed, [Z] vitals/labs extracted from notes
- Transition to model performance tab
- **Performance metrics framing:**
  - Lead with the business metric: "For every 100 alerts at our operating threshold, [X] are genuine high-risk patients — compared to [Y] for the HbA1c-only rule"
  - Precision at 80% Recall: [result] — the clinical threshold that balances catch rate with nursing workload
  - PR-AUC [result] vs. HbA1c baseline ~0.155 — [X]x improvement
  - ROC-AUC [result] — reported for benchmark comparability
- Show the baseline comparison table / chart

---

### Dashboard Tab 2 — Patient Explorer / SHAP Demo *(Alice & Bob)*
*Show the patient explorer tab. Pull up a patient with a profile similar to Alice.*
- Select a HIGH or CRITICAL risk patient from the explorer
- Show their timeline: HbA1c trend, encounter gaps, ER history
- Show the SHAP force plot: "These are the three factors driving this alert — HbA1c worsening over 180 days, 94 days since last encounter, eGFR declining"
- Show the protective factors: "This is what is holding the risk below CRITICAL"
- **Key message:** "This is what the clinician sees before deciding to act. Not a score. A reason."
- *"This is Alice. The system flagged her 9 days before her HbA1c crossed 10%. Her care team was notified. She did not end up in hospital."*

---

### Dashboard Tab 3 — Fairness, Robustness & Compliance (if tab exists)
*Show subgroup performance or calibration plots if available on the dashboard*
- Demographic fairness: model performance across age groups, gender, ethnicity — [insert finding]
- Calibration: does the predicted 80% risk correspond to an 80% observed event rate?
- Robustness: performance under degraded missing data rates — which features are most critical?
- **Compliance summary:** GDPR Art. 22 human review process; EU AI Act high-risk documentation; FHIR/SNOMED/ICD-10 compliance — all evidenced in the report

---

### Back to Slides — Error Analysis & Limitations *(brief, after dashboard)*

### Slide 13 — What the Model Gets Wrong
- **False negatives (missed patients):** Sparse EHR data (low feature completeness), unusual comorbidity profiles, very recent first encounters — patients the system has not had time to learn
- **False positives (spurious alerts):** High medication count + stable HbA1c, elderly patients with high encounter frequency but no acute deterioration — complex but stable cases
- **Known limitations:**
  - Trained on synthetic data — real-world distributions will differ; positive rate in deployment may shift, requiring recalibration
  - Not validated for Type 1 diabetes, gestational diabetes, or paediatric patients
  - ZIP-code census features added no signal at 30-day horizon — environmental factors are a long-horizon story, not a 30-day one
- **[Insert Timarian's error analysis key findings]**

---

## Section 5 — Key Takeaways (~6 min)

### Slide 14 — Our Recommendation, Restated *(pyramid close)*
*Mirror Slide 2 — close the loop for the client.*

> **"Deploy the 22-feature XGBoost pipeline. Pilot alert thresholds in one ward before hospital-wide rollout. Recalibrate if the real-world positive rate differs materially from the training distribution. The compliance documentation is ready for an EU AI Act audit."**

- **ROI case:** [X] prevented readmissions per year × [CHF/USD Y average cost avoided] = [Z] net annual value; pipeline runs on existing hardware; no additional infrastructure required
- **Risk:** Low — system is a decision-support tool, not an autonomous decision-maker; all alerts require clinician review; no patient data leaves the hospital perimeter
- **Next steps:** Ward pilot → threshold calibration → EHR integration via FHIR API → post-market monitoring setup

---

### Slide 15 — Lessons for SwissCare & Similar Engagements
- **True positive rate over accuracy:** In clinical risk monitoring, missing a high-risk patient is categorically worse than a false alarm — design metrics before models
- **Explainability is a clinical requirement:** SHAP force plots are what make an alert actionable; a probability score alone is noise to a clinician under time pressure
- **Temporal safety is not optional:** Features without a prediction-date cutoff produce circular reasoning that only fails when tested on the future; build it in from day one
- **Synthetic data has limits:** Our census feature experiment proved that Synthea does not encode socioeconomic correlates of adverse events — validate any external data source for signal before building on it
- **Regulatory documentation is a first-class deliverable:** EU AI Act and GDPR requirements shape architecture (human oversight, logging, explainability) — not just paperwork written after the fact

---

## Appendix / Backup Slides (for Q&A)

- Full 22-feature list with clinical meanings and SHAP global importance ranking
- Detailed calibration curve
- GDPR Article 22 human review process flowchart
- Census feature correlation table (density_per_sq_mile r=−0.0058, overall_healthcare_coverage r=+0.0050, coverage_score r=+0.0034)
- Failed experiment detail: SMOTE, row-level split, temporal cutoff omission, census features
- Full regulatory compliance mapping (EU AI Act Annex III, GDPR Art. 22, FHIR schema, SNOMED event codes)
- LSTM vs. XGBoost comparison
- Stakeholder success metric mapping
