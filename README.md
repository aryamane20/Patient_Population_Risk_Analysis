# Healthcare Learning - Healthcare AI Portfolio

A progressive set of healthcare data and machine-learning projects, built while
learning to work with clinical data end to end: pulling from FHIR, structuring
free-text notes, running population-health analytics, and finally training and
explaining a readmission-risk model behind a role-based dashboard.

Every project uses **public, sandbox, or synthetic data** and applies real-world
practices: de-identification before analysis, honest (non-inflated) metrics, and
fairness auditing.

---

## What is in this repo

All work lives under `healthcare-ai-projects/`. There are four distinct
projects, each a step up in complexity:

| Project | Folder | Dataset | What it does |
|---------|--------|---------|--------------|
| **1. FHIR Data Pipeline** | `fhir_pipeline/` | HAPI FHIR Public Sandbox | Extracts patients from a live FHIR API, flattens the resources, engineers features, de-identifies, validates data quality, and applies a rules-based risk score. |
| **2. Clinical NLP Pipeline** | `clinical_nlp/` | Synthetic sample discharge summary | Turns free-text discharge summaries into structured data: diagnoses (with negation/assertion), medications, social determinants, referrals, and ML-ready features. |
| **3. Population Health Analytics** | `population_analytics/` | Synthetic simulated population (~10k) | Simulates a patient population, risk-stratifies it, computes HEDIS quality measures, finds care gaps, and builds a care-coordinator worklist, with a Streamlit dashboard. |
| **4. Patient Risk Stratification Dashboard** | `patient_risk_stratification_dashboard/` | UCI Diabetes 130-US Hospitals (#296) | A full ML system: trains readmission-risk models on real data, explains predictions with SHAP, audits fairness, tunes the decision threshold, and serves a role-based dashboard with AI-assisted care plans. |

---

## 1. FHIR Data Pipeline

**Folder:** `healthcare-ai-projects/fhir_pipeline/` (`main.py` + `pipeline_functions.py`)

An ingestion pipeline against the **HAPI FHIR public sandbox** (`hapi.fhir.org`).
It calls the FHIR API, flattens `Patient/Condition/Encounter/Observation`
resources into a patient-level table, engineers features, applies HIPAA-style
**de-identification** (hashing identifiers, generalizing dates), runs
**data-quality validation**, and computes a **rules-based risk score**. In
production this same interface would point at an Epic/Cerner FHIR API with SMART
on FHIR auth.

- **Dataset:** HAPI FHIR Public Sandbox (`hapi.fhir.org`), live public FHIR test server.
- **Run:** `python3 fhir_pipeline/main.py`
- **Output:** versioned `patient_features_v###.csv` (plus a `_LATEST` copy) in
  `fhir_pipeline/outputs/`.

## 2. Clinical NLP Pipeline

**Folder:** `clinical_nlp/clinical_nlp_pipeline.py`

Processes unstructured clinical discharge summaries and extracts diagnoses
(confirmed / historical / negated / possible), medications, social determinants
of health, referrals, and follow-up requirements as structured, ML-ready
features. Uses **scispaCy clinical NER plus a simplified ConText algorithm** for
negation and assertion, with a **pure-regex fallback** so it still runs when no
scispaCy model is installed.

- **Dataset:** a synthetic sample discharge summary (an example clinical note
  embedded in the script; no external dataset required).
- **Run:** `python3 clinical_nlp/clinical_nlp_pipeline.py`

## 3. Population Health Analytics

**Folder:** `population_analytics/`

A population-health platform that generates a realistic simulated patient panel
and then: risk-stratifies it (predictive plus acuity-based), calculates **HEDIS**
quality measures, identifies **care gaps**, prioritizes panel management, produces
a **daily care-coordinator worklist**, and analyzes disease-specific cohorts and
outreach effectiveness.

- **Dataset:** a synthetic simulated patient population (~10,000 patients
  generated in-code with realistic distributions; no external dataset required).
- **Run pipeline:** `python3 population_analytics/run_population_health.py`
- **Dashboard:** `streamlit run population_analytics/population_dashboard.py`
- **Outputs:** `population_risk_stratified.csv`, `hedis_measures_summary.csv`,
  `care_gaps.csv`, `daily_worklist.csv`, `population_health_summary.json`.

## 4. Patient Risk Stratification Dashboard

**Folder:** `patient_risk_stratification_dashboard/`

The most advanced project: a production-minded ML system that predicts **30-day
readmission risk** on the real, de-identified UCI Diabetes 130-US Hospitals
dataset (~70k de-duplicated patients). It converts each probability into a 0-100
score and a High/Medium/Low tier, explains every prediction with **SHAP**, audits
**fairness** across demographic subgroups, and includes a **decision-threshold
tuner** (choose the operating point by the cost of a miss).

On top of the pipeline sits a **role-based Streamlit dashboard** (Care
Coordinator / Clinical Lead / ML Engineer). The coordinator view centers on a
**Today's Worklist**: a recurring outreach queue on a per-tier cadence with a
tick-off contact log, so patients cycle back for follow-up automatically.
**Care plans** are generated by Claude when an API key is present, with a
rule-based fallback otherwise.

- **Dataset:** UCI Diabetes 130-US Hospitals (UCI ML Repository #296, CC BY 4.0),
  101,766 encounters de-duplicated to ~70,000 patients.
- **Run:** see the project's own
  [README](healthcare-ai-projects/patient_risk_stratification_dashboard/README.md)
  for the full pipeline and dashboard instructions.
- **Champion model:** XGBoost, AUROC ~0.65 (honest, in the published range for
  this dataset).

---

## Tech stack

- **Language:** Python 3.12
- **Data / ML:** pandas, numpy, scikit-learn, XGBoost, SHAP
- **NLP:** scispaCy / spaCy (with regex fallback)
- **Interop:** FHIR (HAPI public sandbox), HL7 ADT concepts for the live-data path
- **Apps:** Streamlit dashboards
- **AI:** Anthropic Claude for care-plan generation (optional)

---

## Repository layout

```
healthcare-ai-projects/
  fhir_pipeline/               FHIR data pipeline (project 1)
    main.py                    entry point
    pipeline_functions.py      FHIR extraction, de-identification, rules risk score
    outputs/                   patient_features_*.csv (versioned outputs)
  clinical_nlp/                Clinical NLP pipeline (project 2)
  population_analytics/        Population health platform + dashboard (project 3)
  patient_risk_stratification_dashboard/
                               Readmission-risk ML system + dashboard (project 4)
```

---

## Notes on data and ethics

- No protected health information is used. Sources are a public FHIR sandbox,
  synthetic population data, and a de-identified public research dataset
  (UCI #296, CC BY 4.0).
- Metrics are reported honestly rather than inflated; the readmission model's
  AUROC around 0.65 matches the published literature for that dataset.
- Fairness auditing (subgroup performance and gaps) is built in, not an
  afterthought.
