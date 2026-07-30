"""
fairness.py - subgroup performance + bias metrics (feeds the Day-27 panel).

For each protected attribute (race, gender, age band) we report per-subgroup:
  - n, positive (base) rate
  - AUROC (ranking quality within the subgroup)
  - recall / TPR  and  FNR = 1 - TPR  (a high FNR = high-risk patients MISSED,
    the most clinically dangerous error for a safety-net tool)
  - selection rate (fraction flagged positive)

and two standard fairness gaps across the subgroups:
  - equal-opportunity difference = max(TPR) - min(TPR)
  - demographic-parity difference = max(selection_rate) - min(selection_rate)

Auditing the model (not just shipping it) is the master's-level differentiator.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config

# subgroups smaller than this are pooled/flagged as statistically unreliable
MIN_SUBGROUP_N = 100

# non-informative subgroup labels dropped from the audit (not an actionable group)
EXCLUDED_SUBGROUPS = {"Unknown", "Unknown/Invalid", "?", "None", "nan", ""}


def _age_band(age_midpoint: float) -> str:
    if pd.isna(age_midpoint):
        return "Unknown"
    if age_midpoint < 50:
        return "<50"
    if age_midpoint < 70:
        return "50-69"
    return "70+"


def subgroup_metrics(df: pd.DataFrame, group_col: str,
                     y_col: str = "y_true", proba_col: str = "probability",
                     pred_col: str = "y_pred") -> pd.DataFrame:
    """Per-subgroup performance table for one protected attribute."""
    rows = []
    for value, g in df.groupby(group_col):
        n = len(g)
        y = g[y_col].to_numpy()
        proba = g[proba_col].to_numpy()
        pred = g[pred_col].to_numpy()
        pos = int(y.sum())
        # AUROC undefined if a subgroup has only one class
        auroc = round(float(roc_auc_score(y, proba)), 4) if 0 < pos < n else np.nan
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        rows.append({
            "attribute": group_col,
            "subgroup": str(value),
            "n": n,
            "reliable": n >= MIN_SUBGROUP_N,
            "positive_rate": round(pos / n, 4) if n else np.nan,
            "auroc": auroc,
            "recall_tpr": round(recall, 4) if recall == recall else np.nan,
            "fnr": round(1 - recall, 4) if recall == recall else np.nan,
            "selection_rate": round(float((pred == 1).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def fairness_gaps(table: pd.DataFrame) -> dict:
    """Equal-opportunity and demographic-parity gaps over reliable subgroups."""
    rel = table[table["reliable"]]
    tpr = rel["recall_tpr"].dropna()
    sel = rel["selection_rate"].dropna()
    return {
        "equal_opportunity_diff": round(float(tpr.max() - tpr.min()), 4) if len(tpr) > 1 else 0.0,
        "demographic_parity_diff": round(float(sel.max() - sel.min()), 4) if len(sel) > 1 else 0.0,
        "worst_fnr_subgroup": (
            rel.loc[rel["fnr"].idxmax(), "subgroup"] if rel["fnr"].notna().any() else None
        ),
        "worst_fnr": (
            round(float(rel["fnr"].max()), 4) if rel["fnr"].notna().any() else None
        ),
    }


def build_report(verbose: bool = True) -> dict:
    """Assemble the demographics + predictions frame and compute all bias tables."""
    feats = pd.read_csv(config.DATA_PROCESSED / "features.csv")
    scores = pd.read_csv(config.OUTPUTS_DIR / "patient_risk_scores.csv")

    merged = feats.merge(
        scores[["patient_id", "probability", "predicted_label"]],
        left_on=config.PATIENT_ID_COL, right_on="patient_id", how="inner",
    )
    merged = merged.rename(columns={config.TARGET_COL: "y_true",
                                    "predicted_label": "y_pred"})
    merged["age_band"] = merged["age_midpoint"].map(_age_band)

    report = {"tables": {}, "gaps": {}}
    for attr in ("race", "gender", "age_band"):
        if attr not in merged.columns:
            continue
        # drop non-actionable subgroups (Unknown / Invalid) before auditing
        subset = merged[~merged[attr].astype(str).isin(EXCLUDED_SUBGROUPS)]
        table = subgroup_metrics(subset, attr)
        report["tables"][attr] = table.to_dict(orient="records")
        report["gaps"][attr] = fairness_gaps(table)
        if verbose:
            print(f"\n=== {attr} ===")
            print(table.to_string(index=False))
            print("gaps:", report["gaps"][attr])

    out = config.OUTPUTS_DIR / "fairness_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    if verbose:
        print(f"\nSaved fairness report -> {out}")
    return report


if __name__ == "__main__":
    build_report()
