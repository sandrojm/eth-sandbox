# EHR Patient Monitoring System

An educational project for building an **agentic EHR patient monitoring system** with automated diabetes risk assessment. Students transform synthetic Electronic Health Records (EHR) data into standardized patient profiles and implement predictive models for 30-day adverse event prediction.

---

## Project Overview

### Business Context

Healthcare organizations face critical challenges with patient data:

| Challenge | Impact |
|-----------|--------|
| **Data Fragmentation** | Patient data stored in unstructured formats hinders clinical decision-making |
| **Compliance Complexity** | Complex regulatory landscape requires specialized technical expertise |
| **Care Coordination** | Poor information sharing leads to redundant testing and suboptimal care |

This project addresses these challenges by building an end-to-end pipeline that:
1. Transforms unstructured EHR data into standardized patient profiles
2. Extracts structured information from clinical notes
3. Predicts patient risk for adverse events within 30 days
4. Implements an agentic monitoring system for automated alerts

### Learning Objectives

- **Data Engineering**: Clean, transform, and standardize messy healthcare data
- **NLP/Text Extraction**: Extract structured data from unstructured clinical notes using regex and dictionary-based methods
- **Feature Engineering**: Create clinically meaningful features with proper temporal handling
- **ML Classification**: Build risk prediction models handling class imbalance and patient-level data splitting
- **Agentic Systems**: Design autonomous monitoring systems for healthcare contexts

---

## Project Structure

```
ehr_project/
├── README.md
├── week_1/
│   ├── notebooks/
│   │   ├── session_0_setup_sanity_check.ipynb
│   │   ├── session_3_part_1_data_understanding.ipynb
│   │   ├── session_3_part_2_cleaning_transformation.ipynb
│   │   └── week_1_hw.ipynb
│   ├── src/                     # Source code modules
│   │   ├── profiles/            # PatientProfile classes
│   │   ├── data_processing/     # Extraction & cleaning functions
│   │   └── helpers.py
│   ├── data/                    # Data folder
│        ├── week_1/             # Data for week_1
│        │    ├──raw/
│        │    ├──cleaned_data/
│        │    └──processed_data/ 
│        └── other weeks...
│ 
├── week_2/                      # Feature Engineering & Classification
├── week_3/                      # Model Development & Evaluation
├── week_4/                      # Agentic System & Monitoring
└── week_5/                      # Analysis & Deployment
```

---

## Week 1: Data Understanding & Extraction

### Session 0: Setup Sanity Check
Verify your environment is correctly configured and all data files are accessible.

### Session 3, Part 1: Data Understanding
- Load and explore 7 EHR tables (patients, encounters, conditions, medications, observations, procedures, allergies)
- Analyze patient demographics and encounter patterns
- Assess data sufficiency for ML model building
- Introduction to the `PatientProfile` class

### Session 3, Part 2: Data Cleaning & Transformation
- Handle data quality issues (duplicates, orphaned records, invalid values)
- Clean text data (HTML artifacts, OCR errors)
- Save cleaned data for downstream processing

### Homework: Clinical Note Extraction
- Build regex patterns for extracting vitals, labs, and medications from clinical notes
- Create extraction pipeline for batch processing
- Merge extracted data with original CSV tables
- Perform data quality assessment

---

## Data Model

The Synthea dataset follows a relational structure:

```
PATIENTS (root)
└── Has many ENCOUNTERS
    ├── Has many CONDITIONS
    ├── Has many MEDICATIONS
    ├── Has many OBSERVATIONS
    └── Has many PROCEDURES
```

### Key Tables

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `patients.csv` | Patient demographics | ID, BIRTHDATE, GENDER, RACE |
| `encounters.csv` | Clinical visits | ID, DATE, PATIENT, DESCRIPTION |
| `conditions.csv` | Diagnoses | PATIENT, ENCOUNTER, CODE, DESCRIPTION |
| `medications.csv` | Prescriptions | PATIENT, ENCOUNTER, DESCRIPTION, START, STOP |
| `observations.csv` | Labs & vitals | PATIENT, ENCOUNTER, VALUE, UNITS |
| `procedures.csv` | Medical procedures | PATIENT, ENCOUNTER, DATE, DESCRIPTION |
| `allergies.csv` | Known allergies | PATIENT, DESCRIPTION |

### Clinical Notes
Approximately 25% of encounters exist only in unstructured clinical notes. These notes contain:
- Vital signs (BP, weight, SpO2, temperature)
- Lab values (HbA1c, glucose, creatinine, eGFR)
- Medication mentions
- Clinical observations

---

## Key Concepts

### PatientProfile Class
Consolidates scattered EHR data into a unified object per patient:
- Demographics, conditions, medications, observations
- Chronologically sorted timelines
- Diabetes-specific tracking (HbA1c, diabetes type)

### ExtendedPatientProfile Class
Extends PatientProfile with time-series capabilities:
- `get_daily_features(cutoff_time)` - Compute features at any point in time
- Temporal safety to prevent data leakage
- Risk labeling for 30-day adverse events

### Temporal Safety
All feature engineering must respect temporal boundaries:
```python
# WRONG: Uses future data (data leakage!)
def get_max_hba1c(profile):
    return max(r['value'] for r in profile.hba1c_timeline)

# CORRECT: Only uses data before cutoff_time
def get_max_hba1c(profile, cutoff_time):
    past = [r for r in profile.hba1c_timeline if r['date'] < cutoff_time]
    return max(r['value'] for r in past) if past else None
```

---

## Getting Started

### Prerequisites

All required libraries (pandas, numpy, scikit-learn, matplotlib, seaborn) are pre-installed on Google Colab.

### Running on Google Colab
1. Mount Google Drive
2. Clone or upload the repository to your Drive
3. Update `REPO_PATH` in each notebook to point to your repository location
4. Download the dataset using the provided URL

### Data Setup
1. Obtain the dataset URL from your instructor
2. Run the data download cell in `session_0_setup_sanity_check.ipynb`
3. Verify all 7 CSV files are present in `week_1/data/`

---

## Evaluation Criteria

### Data Extraction (Week 1)
- Completeness of extracted data from clinical notes
- Accuracy of regex pattern matching
- Proper handling of edge cases and missing data

### Risk Classification (Week 2+)
- AUROC for readmission prediction
- Precision-Recall curves (due to class imbalance)
- Demographic parity across patient subgroups

### Agentic System (Week 4+)
- Alert accuracy and timeliness
- False positive/negative rates
- Cost-benefit analysis

---

## Important Notes

### Data Quality Issues (Intentional)
The Week 1 dataset contains realistic data quality issues that students must handle:
- Duplicate patient records
- Orphaned clinical notes
- HTML artifacts in text fields
- OCR errors in scanned documents

### Patient-Level Splitting
When building ML models, always split data at the patient level (not row level) to prevent data leakage from correlated observations within the same patient.

### Class Imbalance
High-risk events are relatively rare. Use techniques like SMOTE, class weighting, or threshold adjustment to handle imbalanced classes.

---

## Resources

- [HL7 FHIR Documentation](https://www.hl7.org/fhir/)
- [Synthea Patient Generator](https://synthetichealth.github.io/synthea/)
- [LOINC Lab Codes](https://loinc.org/)

---

## License

This project is for educational purposes only. The synthetic patient data is generated using Synthea and does not represent real patients.
