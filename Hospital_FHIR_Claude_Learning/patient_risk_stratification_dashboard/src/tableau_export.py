"""
tableau_export.py - build denormalized, BI-friendly CSVs for Tableau.

Tableau works best with a single wide "one row per patient" fact table (so you can
drag any dimension onto a shelf) plus a few small aggregate tables. This module
joins the model outputs to the engineered features and writes everything to
outputs/tableau/.

Files produced:
  patient_level.csv   one row per patient: score, tier, factors, prediction vs
                      actual, demographics, diagnosis groups, utilization counts.
                      -> the main fact table for Tableau.
  tier_summary.csv    patients / mean score / % of panel per tier.
  fairness_long.csv   tidy subgroup metrics (attribute, subgroup, AUROC, FNR, ...).
  model_metrics.csv   LR vs GBM comparison (one row per model).
"""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .fairness import _age_band
from .risk_scoring import percent_of_total

TABLEAU_DIR = config.OUTPUTS_DIR / "tableau"

# clinical/demographic dimensions worth exposing to Tableau
_DIMENSIONS = [
    "race", "gender", "medical_specialty_grp",
    "admission_type_grp", "admission_source_grp", "discharge_disposition_grp",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "a1c_result", "max_glu_serum_grp", "insulin",
]
_MEASURES = [
    "age_midpoint", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "num_chronic_conditions", "num_prior_admissions", "num_meds_changed",
    "num_meds_on", "change_flag", "diabetes_med_flag",
]


def build_patient_level() -> pd.DataFrame:
    feats = pd.read_csv(config.DATA_PROCESSED / "features.csv")
    scores = pd.read_csv(config.OUTPUTS_DIR / "patient_risk_scores.csv")

    merged = scores.merge(
        feats, left_on="patient_id", right_on=config.PATIENT_ID_COL, how="left",
    )
    merged["age_band"] = merged["age_midpoint"].map(_age_band)
    merged = merged.rename(columns={config.TARGET_COL: "actual_readmit_30d"})
    merged["prediction_correct"] = (
        merged["predicted_label"] == merged["actual_readmit_30d"]
    ).astype(int)
    # outcome quadrant for confusion-matrix style views in Tableau
    def _quadrant(r):
        if r["predicted_label"] == 1 and r["actual_readmit_30d"] == 1:
            return "True Positive"
        if r["predicted_label"] == 1 and r["actual_readmit_30d"] == 0:
            return "False Positive"
        if r["predicted_label"] == 0 and r["actual_readmit_30d"] == 1:
            return "False Negative"
        return "True Negative"
    merged["outcome_quadrant"] = merged.apply(_quadrant, axis=1)

    cols = (
        ["patient_id", "risk_score", "tier", "probability", "predicted_label",
         "actual_readmit_30d", "prediction_correct", "outcome_quadrant",
         "top_3_risk_factors", "age_band"]
        + [c for c in _DIMENSIONS if c in merged.columns]
        + [c for c in _MEASURES if c in merged.columns]
    )
    return merged[cols]


def build_tier_summary(patient_level: pd.DataFrame) -> pd.DataFrame:
    order = ["High", "Medium", "Low"]
    g = (patient_level.groupby("tier")
         .agg(patients=("risk_score", "size"),
              mean_score=("risk_score", "mean"),
              mean_probability=("probability", "mean"),
              actual_readmit_rate=("actual_readmit_30d", "mean"))
         .reindex(order).dropna(how="all"))
    g["mean_score"] = g["mean_score"].round(1)
    g["mean_probability"] = g["mean_probability"].round(4)
    g["actual_readmit_rate"] = g["actual_readmit_rate"].round(4)
    g["pct_of_panel"] = percent_of_total(g["patients"])
    return g.reset_index()


def build_fairness_long() -> pd.DataFrame:
    path = config.OUTPUTS_DIR / "fairness_report.json"
    if not path.exists():
        return pd.DataFrame()
    report = json.loads(path.read_text())
    frames = []
    for attr, records in report.get("tables", {}).items():
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_model_metrics() -> pd.DataFrame:
    path = config.MODELS_DIR / "metrics.json"
    if not path.exists():
        return pd.DataFrame()
    metrics = json.loads(path.read_text())
    rows = []
    for name, m in metrics.get("models", {}).items():
        rows.append({
            "model": name,
            "is_champion": name == metrics.get("champion"),
            "auroc": m.get("auroc"), "pr_auc": m.get("pr_auc"),
            "recall": m.get("recall"), "precision": m.get("precision"),
            "f1": m.get("f1"), "brier": m.get("brier"),
        })
    return pd.DataFrame(rows)


def run(verbose: bool = True) -> dict:
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

    patient_level = build_patient_level()
    outputs = {
        "patient_level.csv": patient_level,
        "tier_summary.csv": build_tier_summary(patient_level),
        "fairness_long.csv": build_fairness_long(),
        "model_metrics.csv": build_model_metrics(),
    }
    for name, df in outputs.items():
        if df is None or df.empty and name != "patient_level.csv":
            if verbose:
                print(f"  (skipped {name} - source not found)")
            continue
        df.to_csv(TABLEAU_DIR / name, index=False)
        if verbose:
            print(f"  wrote {name:20s} {df.shape[0]:>6,} rows x {df.shape[1]} cols")

    if verbose:
        print(f"\nTableau CSVs -> {TABLEAU_DIR}")
        print("\nMain fact table columns (patient_level.csv):")
        print("  " + ", ".join(patient_level.columns))
    return outputs


if __name__ == "__main__":
    run()
