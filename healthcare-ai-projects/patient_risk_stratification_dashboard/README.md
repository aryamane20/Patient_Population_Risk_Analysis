# Patient Risk Stratification Dashboard

Predicts **30-day hospital readmission risk** for discharged patients, turns each
probability into a **0-100 risk score** and a **High / Medium / Low tier**, and
explains *why* each patient is at risk (per-patient top-3 factors via SHAP) so
care coordinators know who to call and what to address.

> **What "risk" means here:** the probability of one specific event, readmission
> within 30 days, not a vague "how sick" number. The model learns from ~70k past
> discharges where the outcome is known and applies that to score patients.

The project has three layers that build on each other:

1. **Backend pipeline** - data ingest, cleaning, feature engineering, model
   training, risk scoring, explainability, and a bias audit. Produces
   `outputs/patient_risk_scores.csv`.
2. **Streamlit dashboard** (`app.py`) - a role-based UI on top of those outputs.
3. **AI-assisted care plans** (`src/care_plan.py`) - Claude-generated or
   rule-based intervention checklists per patient.

---

## Results (held-out test split, de-duplicated cohort)

| Model | AUROC | PR-AUC | Recall | Precision | Brier |
|-------|-------|--------|--------|-----------|-------|
| Logistic Regression (baseline) | 0.640 | 0.150 | 0.527 | 0.133 | 0.232 |
| **XGBoost (champion)** | **0.648** | **0.157** | 0.531 | 0.138 | 0.215 |

- Cohort: **69,990** patients (first encounter each), **positive rate 8.98%**.
- AUROC around 0.65 is **in the published literature range (~0.64-0.70)** for this
  dataset (Strack et al. 2014). We report this honestly rather than inflating it
  by keeping duplicate encounters; a model that scored far higher here would
  signal a leakage bug, not skill.
- Champion selected by **PR-AUC** (imbalance-aware), which is roughly 1.75x the
  0.09 base rate.

---

## Architecture - production-ready by design

The model depends only on a **`DataSource.load()`** contract and a shared
**feature schema**. Swapping the data source (public CSV to a live FHIR/Epic
feed) means writing one class, not touching the model.

```
raw source -> ingest.py -> clean.py -> features.py -> model.py -> risk_scoring.py
 (CSV now,     DataSource   dedup/      feature       LR + GBM    0-100 score
  FHIR later)  interface    leakage     "contract"    Pipeline    + tiers
                            guards                        |
                                                          |-> explain.py   (SHAP: global + per-patient top-3)
                                                          |-> export.py    (patient_risk_scores.csv, OOF-scored)
                                                          |-> fairness.py  (subgroup AUROC / FNR / gaps)
                                                          |-> threshold.py (decision-threshold sweep + cost tuner)
```

### Three data tiers (documented live-data path)
1. **Now (real, historical):** Diabetes 130-US Hospitals via `CSVSource`.
2. **Near-term (live clinical):** a `FHIRSource` adapter maps
   `Patient/Condition/Encounter/Observation` to the same feature schema; a nightly
   batch scores the active panel; real-time admits arrive via an HL7 ADT
   (A01/A03) feed.
3. **Enterprise (production EHR):** Epic Caboodle as the batch source, HL7 ADT for
   streaming, de-identification before any ML store.

---

## Dataset

- **Source:** UCI ML Repository #296, Diabetes 130-US Hospitals, 101,766
  encounters, ~47 features, 1999-2008, 130 US hospitals.
- **License:** CC BY 4.0 (safe for a public portfolio, with attribution).
- **Target:** `readmitted` mapped to binary `readmit_30d` = 1 if `<30` days else 0.
- **Origin paper (cited):** Strack et al., 2014, *BioMed Research International*,
  also the source of the ICD-9 to disease-group feature mapping.

The zip auto-downloads on first run into `data/raw/` (git-ignored).

---

## How to run

```bash
# 1. Create an isolated environment (do NOT use the anaconda base env)
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# 2. Run the pipeline end-to-end
./.venv/bin/python -m src.ingest          # download + load 101,766 rows
./.venv/bin/python -m src.clean           # -> data/processed/cleaned.csv (69,990 rows)
./.venv/bin/python -m src.features        # -> data/processed/features.csv (26 features)
./.venv/bin/python -m src.model           # train LR + GBM -> models/*.joblib + metrics.json
./.venv/bin/python -m src.export          # -> outputs/patient_risk_scores.csv (all patients, OOF)
./.venv/bin/python -m src.fairness        # -> outputs/fairness_report.json
./.venv/bin/python -m src.threshold       # -> outputs/threshold_sweep.csv (decision tuner)
./.venv/bin/python -m src.tableau_export  # -> outputs/tableau/*.csv (BI-ready)

# 3. Tests
./.venv/bin/python -m pytest tests/ -q

# 4. Launch the dashboard
./.venv/bin/python -m streamlit run app.py
# optional: enable Claude-generated care plans
export ANTHROPIC_API_KEY=sk-ant-...        # otherwise a rule-based engine is used
```

> **Environment note:** use a per-project `.venv`. Installing `shap` into the
> anaconda base env once upgraded numpy to 2.0 and broke other projects; the venv
> isolates these dependencies.

---

## Key design decisions

| Decision | Choice | Why |
|----------|--------|-----|
| De-duplication | Keep **first encounter** per patient | Prevents within-patient leakage (Strack 2014). Drops 101,766 to 69,990, positive rate 11.2% to 8.98%. Honest over inflated. |
| Who gets scored | **Everyone**, via out-of-fold `cross_val_predict` | No patient scored in-sample; headline metrics still from the held-out test. |
| Risk tiers | **Quantile-based** (High >= P80, Med P50-P80, Low < P50) | This data's probabilities cluster low; fixed cutoffs would leave the High list empty. |
| Class imbalance | `class_weight='balanced'` / `scale_pos_weight` | Handles the ~9% positive rate. |
| Intermediate storage | **CSV** (not Parquet) | Avoids a `pyarrow` dependency for a learning project. |

> **Reading the score:** because the champion is imbalance-weighted, the raw
> `probability` is **adjusted for ranking, not a calibrated literal frequency**.
> The 0-100 `risk_score` is a *relative* risk index; the quantile **tiers** are
> what drive care-coordinator action.

---

## Dashboard

`streamlit run app.py` opens a role-based dashboard. A sidebar selector picks the
audience and shows only the relevant tabs, so a care coordinator is not buried in
model diagnostics.

**I am a... (sidebar role selector)**

| Role | Tabs shown |
|------|------------|
| **Care Coordinator** | Today's Worklist, Patient Explorer, Patient Drill-down |
| **Clinical Lead** | Today's Worklist, Population, Bias Monitoring |
| **ML Engineer** | Population, Patient Explorer, Patient Drill-down, Decision Threshold, Model Performance, Bias Monitoring |

**Views**

- **Today's Worklist** - the coordinator's landing page. A recurring outreach
  queue: each patient reappears on a **cadence** by tier (default High every 7
  days, Medium every 30, Low every 90; editable in the UI). Ordered Overdue, then
  Never-contacted, then Due, highest risk first. Tick **Contacted today** and save;
  contacts are written to `contact_log.csv` so patients drop off until their next
  due date and the next patients move up. Includes an optional "only show my first
  N" capacity cap.
- **Patient Explorer** - sortable, filterable risk list with CSV download.
- **Patient Drill-down** - per-patient score, tier, top-3 factors, a per-patient
  SHAP contribution chart (red raises risk, green lowers it), and a one-click care
  plan.
- **Population** - tier counts and risk-score distribution as interactive charts.
- **Decision Threshold** - an interactive tuner: set the cost of a missed
  readmission versus a follow-up call and watch recall, precision, patients
  flagged, and the operating point on a precision-recall curve update live.
- **Model Performance** - LR vs GBM metrics, confusion matrix in plain language,
  and a global driver chart.
- **Bias Monitoring** - subgroup AUROC / false-negative rate plus
  equal-opportunity and demographic-parity gaps.

The sidebar **tier and score filters** apply to Today's Worklist, Patient
Explorer, and Patient Drill-down. The Population tab is intentionally a
whole-panel overview.

---

## AI-assisted care plans

`src/care_plan.py` turns a patient's top risk factors into a prioritized 30-day
intervention checklist. If `ANTHROPIC_API_KEY` is set, the plan is generated by
**Claude** (structured JSON prompt); otherwise a deterministic **rule-based
engine** is used, so the dashboard always works offline. The model id is
overridable via `CLAUDE_MODEL`.

---

## Fairness / bias monitoring

`src/fairness.py` reports per-subgroup (race, gender, age band) AUROC, recall,
**false-negative rate** (missed high-risk patients, the dangerous error), and
selection rate, plus equal-opportunity and demographic-parity gaps. Non-actionable
subgroups (for example "Unknown") are excluded. This feeds the Bias Monitoring
view. Auditing the model, not just shipping it, is the point.

---

## Outputs

- `outputs/patient_risk_scores.csv` - `patient_id, risk_score, tier,
  top_3_risk_factors` (plus `probability`, `predicted_label`).
- `outputs/global_importance.csv` - plain-language global feature drivers.
- `outputs/shap_global_importance.png` - global SHAP figure.
- `outputs/threshold_sweep.csv` - precision/recall/flagged across thresholds.
- `outputs/fairness_report.json` - subgroup AUROC / FNR and fairness gaps.
- `models/champion_pipeline.joblib`, `models/metrics.json`.
- `outputs/tableau/` - BI-ready CSVs (`python -m src.tableau_export`):
  - `patient_level.csv` - wide one-row-per-patient fact table (score, tier,
    prediction vs actual, `outcome_quadrant`, demographics, diagnosis groups,
    utilization). The main table for Tableau.
  - `tier_summary.csv`, `fairness_long.csv`, `model_metrics.csv` - aggregate
    tables.

`contact_log.csv` (outreach history for the worklist) is created at the project
root at runtime and is git-ignored.

---

## Project layout

```
app.py        Streamlit dashboard (role-based views)
src/          config, ingest, clean, features, model, risk_scoring, explain,
              export, fairness, threshold, tableau_export, care_plan
tests/        pytest unit tests for clean.py + features.py
notebooks/    01_data_pipeline.ipynb  (narrated, graded deliverable)
data/         raw/ (git-ignored downloads) + processed/
models/       serialized pipeline + metrics.json
outputs/      scores CSV, SHAP figure, fairness report, threshold sweep, tableau/
docs/         PROJECT_PLAN.md
```

---

## Sources
- [UCI - Diabetes 130-US Hospitals (#296)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- [Strack et al. 2014 (dataset origin paper)](https://www.hindawi.com/journals/bmri/2014/781670/)
