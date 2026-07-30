# Patient Risk Stratification Dashboard - Capstone Plan (Day 25 + Roadmap)

## Concept primer (plain language)
"Risk" = the probability of **one specific bad event within a time window**, not a vague "how sick" number. Here the event is **30-day readmission**: a discharged patient having to return to the hospital within 30 days (a costly, CMS-penalized signal that something went wrong). The model learns from ~101k **past** discharges where we already know who bounced back, finds which patient traits (prior admissions, # diagnoses, labs, age…) predicted it, and outputs a **probability → 0–100 risk score** for each patient. The Diabetes 130 cohort is diabetic inpatients; we predict their *readmission*, not their diabetes.

## Context
This is the capstone for the Mindbowser Healthcare AI track: a **Patient Risk Stratification Dashboard** for care coordinators. Day 25 delivers the **backend data pipeline** - the foundation Days 26–27 (Streamlit UI, Claude care plans, bias panel) sit on. The goal is a *master's-level, industry-showcase* pipeline built on **real public data** and architected to be **production-ready** (swap the data source to a live FHIR/Epic feed without rewriting the model).

**Decisions locked with the user:**
- **Dataset:** Diabetes 130-US Hospitals (real, de-identified) → target = **30-day readmission risk**. Risk score = P(readmit ≤30 days) × 100.
- **Models:** Logistic Regression (interpretable baseline) **+** Gradient Boosting (champion), compared by AUROC/PR-AUC.
- **Data stance:** real data now + a fully documented production/live-data path.
- **Scope:** Day 25 detailed + Days 26–27 roadmap (matches the 3-tier architecture diagram).
- **Location:** `patient_risk_stratification_dashboard/` (folder already created).

## The dataset (real, confirmed)
- **Source:** UCI ML Repository #296 - 101,766 encounters, ~47 features, 10 yrs / 130 US hospitals.
- **Direct download (no auth):** `https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip`
- **License:** CC BY 4.0 (free to use with attribution - safe for a public portfolio).
- **Also on Kaggle** (`brandao/diabetes`) via the Kaggle API if preferred.
- **File layout (important):** the zip has ONE patient-level table `diabetic_data.csv` (each row = one encounter, carrying `patient_nbr` + all features + `readmitted` together - so the risk score is just a new column on that same row; no cross-table patient join needed) plus a small dictionary `IDS_mapping.csv` that only translates `admission_type_id`/`discharge_disposition_id`/`admission_source_id` codes to text. In production, the trained model is applied to *our own* patients via the shared **feature schema**, never by key-joining the two different populations.
- **Target:** `readmitted` ∈ {`<30`, `>30`, `NO`} → binary `readmit_30d` = 1 if `<30` else 0 (~11% positive → imbalanced).
- **Canonical reference (cite in the notebook):** Strack et al., 2014, *BioMed Research International* - the paper that published this dataset and the ICD-9→disease-group feature approach. Adds academic credibility.

## Proposed repo structure (inside `patient_risk_stratification_dashboard/`)
```
patient_risk_stratification_dashboard/
├── data/
│   ├── raw/              # downloaded zip + diabetic_data.csv (git-ignored)
│   └── processed/        # cleaned + feature parquet/csv
├── notebooks/
│   └── 01_data_pipeline.ipynb      # the Day-25 deliverable (documents every step)
├── src/
│   ├── config.py         # paths, seed, thresholds, column groups
│   ├── ingest.py         # DataSource interface: CSVSource now, FHIRSource later
│   ├── clean.py          # nulls, '?'→NaN, outliers, dedup, exclusions
│   ├── features.py       # engineered feature builders (the "feature contract")
│   ├── model.py          # sklearn Pipeline: preprocess + LR / GBM, train/eval
│   ├── risk_scoring.py   # proba→0-100 score + High/Med/Low tiers
│   ├── explain.py        # SHAP: global importance + per-patient top-3 factors
│   └── export.py         # final CSV writer
├── models/               # serialized pipeline (joblib) + metrics.json
├── outputs/              # patient_risk_scores.csv, plots, SHAP figures
├── requirements.txt
└── README.md             # architecture, data provenance, how-to-run
```
The notebook drives/imports `src/` functions so logic is both documented (notebook) and reusable/production-grade (modules) - not copy-pasted.

## Reuse from existing projects (do not rebuild)
- **Production FHIR ingestion + de-identification:** `pipeline_functions.py` (`get_patients/get_conditions/get_encounters`, `engineer_features()`, `deidentify_patient()`) - becomes the `FHIRSource` adapter in `ingest.py` for the live-data path.
- **Risk-tier pattern:** `population_analytics/population_health.py` `calculate_risk_scores()` uses `pd.cut` tiering + `np.select` care-management assignment - mirror this for `risk_scoring.py` tiers and Day-26 care levels.
- **NLP feature layer (architecture "NLP Feature Extraction"):** `clinical_nlp/clinical_nlp_pipeline.py` `extract_structured_features()` - the production hook for turning discharge-note text into model features.

## Day 25 - Data pipeline (detailed, executable)

**Step 0 - Environment.** Add `requirements.txt`: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, joblib, jupyter. **Use a dedicated virtualenv `.venv` inside this project - NOT the anaconda base env.** (Correction: an earlier attempt installed `shap` into anaconda base, which upgraded numpy to 2.0 and broke matplotlib for the other projects. A per-project venv isolates these deps.) Create with `/opt/anaconda3/bin/python3 -m venv .venv`, then `./.venv/bin/python -m pip install -r requirements.txt`. Set global `RANDOM_SEED=42`.

**Step 1 - Ingest (`ingest.py`).** Define a small `DataSource` protocol with `load() -> pd.DataFrame`. Implement `CSVSource` that downloads the UCI zip (if absent), unzips, reads `diabetic_data.csv`. Stub `FHIRSource` (documented, wired in production) that calls the Day-14 pipeline. This interface is what makes the pipeline "production-ready" - the model never knows where rows came from.

**Step 2 - Clean (`clean.py`).**
- Replace `'?'` → `NaN`. Drop near-empty columns (`weight` ~97% missing, `payer_code`); keep `medical_specialty` but bucket rare values → `Other`.
- **Exclusions (prevent label leakage / invalid targets):** drop encounters with `discharge_disposition_id` ∈ expired/hospice codes (11,13,14,19,20,21) - those patients cannot be readmitted.
- **De-duplicate:** keep first encounter per `patient_nbr` (repeated patients otherwise leak future info).
- **Outliers:** cap `time_in_hospital`, `num_medications`, `number_*` visit counts at the 99th percentile (IQR/quantile), document counts affected.
- Standardize: age intervals (`[70-80)`) → ordinal midpoint (75); booleans for `change`, `diabetesMed`.

**Step 3 - Feature engineering (`features.py`) - the guideline's five, plus depth.**
- `age_group` (ordinal from the 10-yr buckets).
- `num_chronic_conditions` proxy = `number_diagnoses`.
- `num_prior_admissions` = `number_inpatient + number_emergency + number_outpatient` (prior-year utilization - strongest readmission signal).
- `admission_type` / `admission_source` / `discharge_disposition` → grouped categoricals.
- **ICD-9 diagnosis grouping** (`diag_1/2/3` → circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasm, other) per Strack 2014.
- Lab flags: `A1Cresult`, `max_glu_serum` (measured/high/normal/none).
- Medication churn: count of the 23 drug columns changed (`up`/`down`).
- Output a documented **feature schema** (the "contract") reused by both CSV and FHIR sources.

**Step 4 - Preprocess + split (`model.py`).** Stratified train/test split (80/20, `stratify=y`, seed). Build a single `sklearn.Pipeline` with `ColumnTransformer` (median-impute + scale numerics; most-frequent-impute + one-hot categoricals) so preprocessing is fit on train only (no leakage) and travels with the serialized model.

**Step 5 - Models (`model.py`).**
- **Baseline:** `LogisticRegression(class_weight='balanced', max_iter=1000)`.
- **Champion:** `XGBClassifier` (or `HistGradientBoostingClassifier` fallback) with `scale_pos_weight` for the ~11% positive rate.
- **Evaluate:** AUROC **and** PR-AUC (imbalance-aware), plus a calibration curve and confusion matrix at the chosen threshold. Save `metrics.json`; persist the champion pipeline with `joblib`.

**Step 6 - Risk scoring + tiers (`risk_scoring.py`).** `risk_score = round(predict_proba[:,1] * 100)`. Tiers by quantile (resolved): **High = top 20% (≥P80), Medium = P50–P80, Low = <P50** - see `config.TIER_QUANTILES`. Quantiles avoid arbitrary bins and guarantee a workable High list. Mirror `population_health.py`'s `pd.cut` pattern.

**Step 7 - Explainability (`explain.py`).** `shap.TreeExplainer` on the GBM. Global: summary/bar plot of top drivers. **Per-patient `top_3_risk_factors`:** the 3 features with the largest positive SHAP values for that row, mapped to human-readable labels (e.g. "3 prior inpatient admissions", "HbA1c high"). This is the column care coordinators actually act on.

**Step 8 - Export (`export.py`).** Write `outputs/patient_risk_scores.csv` with exactly: `patient_id, risk_score, tier, top_3_risk_factors` (+ keep `probability`, `predicted_label` as extra columns for the dashboard).

**Step 9 - Notebook (`notebooks/01_data_pipeline.ipynb`).** Narrate every step with markdown: problem framing, data provenance + license, EDA (class balance, missingness, distributions), each cleaning/feature decision with rationale, model comparison table (LR vs GBM AUROC/PR-AUC), SHAP figures, and a "Production & Limitations" section. This *is* the Day-25 grade deliverable.

## Production-ready & real/live-data path (the "can we bring real data in" answer)
**Yes - three concrete tiers, documented in the README + notebook:**
1. **Now (real, historical):** Diabetes 130 via `CSVSource`. Real de-identified inpatient data, runs today.
2. **Near-term (live clinical):** `FHIRSource` adapter reusing `pipeline_functions.py` - pull `Patient/Condition/Encounter/Observation` from a FHIR endpoint, map to the **same feature schema** from Step 3. A nightly batch scores the active panel. Real-time admits arrive via an **HL7 ADT** feed (A01/A03 events) triggering a score.
3. **Enterprise (production EHR):** **Epic Caboodle** (clinical data warehouse) as the batch source; **HL7 ADT** for streaming; de-identify via `deidentify_patient()` before any ML store. Document the FHIR-resource → feature mapping table so the contract is explicit.
4. **Research validation option:** **MIMIC-IV** via PhysioNet (CITI training + credentialed account + DUA, ~1–2 wk lead time) to validate the model on an independent real cohort - noted as an optional stretch, not on the Day-25 critical path.

**MLOps rigor (what makes it "production-ready", not a notebook):** config-driven paths/params, fixed seeds, serialized pipeline artifact, `metrics.json`, structured logging, a few `pytest` unit tests on `clean.py`/`features.py` (schema + null handling), and a `README` run guide. The `DataSource` interface means swapping CSV→FHIR touches one class.

## Fairness / bias monitoring (feeds the Day-27 panel)
Diabetes 130 carries `race`, `gender`, `age` - compute **subgroup AUROC and false-negative rate** per group and fairness gaps (equal-opportunity difference, demographic parity). Surface as a table now; Day 27 renders it as the "Bias Monitoring Panel." Master's-level differentiator: shows you audit the model, not just ship it.

## Full capstone roadmap
- **Day 26 - Streamlit frontend:** population summary (tier counts, KPIs), sortable/filterable risk-patient table, patient drill-down showing `top_3_risk_factors` + SHAP. Reuse dashboard patterns from `population_analytics/population_dashboard.py` (cached loaders, tabs, `use_container_width=True` for Streamlit 1.47).
- **Day 27 - Product layer:** Claude-generated care plans from each patient's top risk factors (Claude API, structured prompt → intervention checklist); render the bias panel; polish docs/README; package for demo.

## Verification (end-to-end, Day 25)
1. `pip install -r requirements.txt`; run `python -m src.ingest` → confirms the CSV downloads and loads 101,766 rows.
2. Run the notebook top-to-bottom under anaconda Python - no errors; class balance, model table, and SHAP plots render.
3. `python -m src.model` trains LR + GBM, prints AUROC/PR-AUC, writes `models/*.joblib` + `metrics.json`. Sanity: GBM AUROC ≈ 0.64–0.70 is the known literature range for this dataset (flag if wildly off - signals leakage).
4. `python -m src.export` produces `outputs/patient_risk_scores.csv`; assert columns `patient_id, risk_score, tier, top_3_risk_factors`, scores in [0,100], tier distribution sane (not all one tier).
5. Spot-check 3 high-risk rows: their `top_3_risk_factors` should be clinically plausible (prior admissions, # diagnoses, discharge disposition).
6. `pytest` passes on `clean.py`/`features.py`.

## Resolved decisions (post-review - agreed, not yet implemented beyond ingest/clean)
1. **De-duplication: ON (keep first encounter per `patient_nbr`).** The academically correct choice (matches Strack 2014); prevents within-patient leakage. Trade-off accepted: drops ~29k repeat encounters (101,766 → 69,990) and lowers the positive rate 11.2% → **8.98%**, which also lowers headline AUROC. We state this honestly rather than inflate it by keeping duplicates.
2. **Who gets scored: EVERYONE (all cleaned patients), via out-of-fold predictions.** Metrics are reported on a held-out test split; the exported risk scores for the full population use `cross_val_predict` (out-of-fold) so training-set patients aren't scored in-sample/optimistically. This gives the Day-26 dashboard a complete patient list with honest scores.
3. **Risk tiers: quantile-based** - High = top 20% of scores (≥P80), Medium = P50–P80, Low = <P50 (see `config.TIER_QUANTILES`). Quantiles guarantee a workable-sized High list for care coordinators regardless of the model's absolute calibration. (Fixed probability cutoffs rejected - this dataset's probabilities cluster low.)
4. **Intermediate storage: CSV, not Parquet.** Parquet needs `pyarrow`/`fastparquet` (not installed); a CSV in `data/processed/` avoids an extra dependency for a learning project.

## Standing notes
- **Score realism:** this dataset's readmission signal is modest (AUROC ~0.65 on de-duplicated data); the notebook states this honestly rather than overclaiming - candor reads as senior, not weak.
- **Kaggle vs UCI:** default to UCI direct download (no auth); offer Kaggle API path in README.
- **Environment:** dedicated `.venv` per project (Correction #1 from review) - never install into anaconda base.

## Build status (Day 25 COMPLETE)
- ✅ `.venv` + `requirements.txt`; ✅ `config.py`; ✅ `ingest.py` (real 101,766-row download verified); ✅ `clean.py` (verified: 69,990 rows, 8.98% positive - now writes CSV per decision #4).
- ✅ `features.py` (26 features; ICD-9 grouping per Strack 2014; documented `FeatureSchema` contract).
- ✅ `model.py` (stratified split, leakage-safe Pipeline, LR + XGBoost; **GBM AUROC 0.648 / PR-AUC 0.157**, in the 0.64–0.70 literature range; champion + `metrics.json` persisted).
- ✅ `risk_scoring.py` (0–100 score + quantile tiers).
- ✅ `explain.py` (SHAP global plot + per-patient top-3 factors, one-hot aggregated to base features).
- ✅ `export.py` (all 69,990 patients scored out-of-fold → `outputs/patient_risk_scores.csv`; tiers 20.9% High / 30.1% Med / 49% Low).
- ✅ `fairness.py` (subgroup AUROC/FNR + equal-opportunity & demographic-parity gaps → `outputs/fairness_report.json`).
- ✅ `notebooks/01_data_pipeline.ipynb` (narrated, executes end-to-end with the `hosp-risk` venv kernel, 0 errors).
- ✅ `tests/` (29 pytest unit tests on `clean.py` + `features.py`, all passing); ✅ `README.md`.

## Days 26–27 COMPLETE
- ✅ **Day 26 - Streamlit dashboard** (`app.py`): KPI row; Population, Patient Explorer (sortable/filterable + CSV download), and Patient Drill-down tabs. Reuses the cached-loader / tabs patterns from `population_analytics/population_dashboard.py`. Verified end-to-end via `streamlit.testing.v1.AppTest` (0 exceptions; migrated off the deprecated `use_container_width` to `width="stretch"`).
- ✅ **Day 27 - Product layer**:
  - `src/care_plan.py`: Claude-generated 30-day care plans (structured JSON prompt) with a **deterministic rule-based fallback** when `ANTHROPIC_API_KEY` is absent or the API errors - the UI always returns a plan. Model id overridable via `CLAUDE_MODEL`.
  - Per-patient **SHAP contribution** chart in the drill-down (`explain.patient_contributions`).
  - **Bias Monitoring Panel** tab rendering `fairness_report.json` (subgroup AUROC/FNR + gaps).
  - **Model Performance** tab (LR vs GBM metrics, confusion matrix, global SHAP figure).
  - `requirements.txt` updated (`streamlit`, `anthropic`, `ipykernel`); README run guide extended.

## Sources
- [UCI - Diabetes 130-US Hospitals (#296)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- [Kaggle - Diabetes 130 (brandao/diabetes)](https://www.kaggle.com/datasets/brandao/diabetes)
- [Strack et al. 2014 (dataset origin paper)](https://www.hindawi.com/journals/bmri/2014/781670/)
- [ODSC - 18 Open Healthcare Datasets, 2025](https://opendatascience.com/18-open-healthcare-datasets-2025-update/)
