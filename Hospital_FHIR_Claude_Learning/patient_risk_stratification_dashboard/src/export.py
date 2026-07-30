"""
export.py - produce the final patient-level deliverable (Step 8).

Scores EVERY patient (plan decision #2) with OUT-OF-FOLD probabilities via
cross_val_predict: each patient is scored by a model that did NOT train on them,
so the full-population risk list is honest (no in-sample optimism). Reported
headline metrics still come from the held-out test split in model.py.

Output: outputs/patient_risk_scores.csv with the required columns
    patient_id, risk_score, tier, top_3_risk_factors
plus probability and predicted_label as extras for the Day-26 dashboard.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .explain import explain
from .model import compute_scale_pos_weight, get_feature_data, make_pipeline
from .risk_scoring import score_frame, tier_summary


def _champion_kind() -> str:
    metrics_path = config.MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text()).get("champion", "gbm")
        except Exception:
            pass
    return "gbm"


def out_of_fold_probabilities(X, y, schema, kind: str) -> np.ndarray:
    """OOF P(readmit) for the full population - no patient scored in-sample."""
    spw = compute_scale_pos_weight(y)
    pipe = make_pipeline(kind, schema, scale_pos_weight=spw)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def build_export(verbose: bool = True) -> pd.DataFrame:
    X, y, ids, schema = get_feature_data()
    kind = _champion_kind()

    if verbose:
        print(f"Scoring {len(X):,} patients out-of-fold with '{kind}' "
              f"({config.CV_FOLDS}-fold)...")
    proba = out_of_fold_probabilities(X, y, schema, kind)

    frame = score_frame(ids, proba)
    frame["predicted_label"] = (frame["probability"] >= 0.5).astype(int)

    # Per-patient explanations from a champion fit on all data (SHAP explains, it
    # does not predict, so a full-data fit is appropriate and gives everyone a row).
    champ_path = config.MODELS_DIR / "champion_pipeline.joblib"
    if champ_path.exists():
        pipe = joblib.load(champ_path)
    else:
        spw = compute_scale_pos_weight(y)
        pipe = make_pipeline(kind, schema, scale_pos_weight=spw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
    if verbose:
        print("Computing per-patient top-3 risk factors (SHAP)...")
    top_factors, shap_matrix, feature_names = explain(pipe, X, make_global_plot=True, k=3)
    frame["top_3_risk_factors"] = top_factors

    # friendly global-driver table for the dashboard (plain-language chart)
    from .explain import global_importance_frame
    gi = global_importance_frame(shap_matrix, feature_names, top_n=12)
    gi.to_csv(config.OUTPUTS_DIR / "global_importance.csv", index=False)

    # decision-threshold sweep (drives the Decision Threshold tuner)
    from .threshold import sweep_thresholds
    sweep_thresholds(proba, y).to_csv(
        config.OUTPUTS_DIR / "threshold_sweep.csv", index=False)

    frame = frame.rename(columns={config.PATIENT_ID_COL: "patient_id"})
    ordered = ["patient_id", "risk_score", "tier", "top_3_risk_factors",
               "probability", "predicted_label"]
    frame = frame[ordered].sort_values("risk_score", ascending=False).reset_index(drop=True)

    out_path = config.OUTPUTS_DIR / "patient_risk_scores.csv"
    frame.to_csv(out_path, index=False)

    if verbose:
        print(f"\nWrote {len(frame):,} scored patients -> {out_path}")
        print("\nTier distribution:")
        print(tier_summary(frame.rename(columns={})).to_string(index=False))
        print("\nTop 3 highest-risk patients:")
        for _, r in frame.head(3).iterrows():
            print(f"  patient {r['patient_id']}  score={r['risk_score']}  "
                  f"tier={r['tier']}  |  {r['top_3_risk_factors']}")
    return frame


if __name__ == "__main__":
    build_export()
