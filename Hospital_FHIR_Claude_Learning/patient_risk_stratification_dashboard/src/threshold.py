"""
threshold.py - choose the flag cut-off by the CLINICAL cost of a miss.

The model outputs a probability; turning it into a yes/no "flag for outreach"
decision requires a threshold. The default 0.5 is rarely right for a safety-net
tool, because a MISSED readmission (false negative) is far costlier than an extra
follow-up call (false positive).

This module sweeps every threshold on honest out-of-fold predictions and picks the
one that minimizes total expected cost:

    total_cost = cost_fn * (missed readmissions) + cost_fp * (false alarms)

Lowering the threshold raises recall (catch more readmissions) at the price of
precision (more false alarms). The right point is a business/clinical decision,
so the dashboard exposes it as a slider; this module provides the math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DEFAULT_COST_FN = 10.0   # a missed readmission is ~10x costlier than one outreach call
DEFAULT_COST_FP = 1.0


def load_oof() -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities + true labels from the exported artifacts."""
    scores = pd.read_csv(config.OUTPUTS_DIR / "patient_risk_scores.csv")
    feats = pd.read_csv(config.DATA_PROCESSED / "features.csv")[
        [config.PATIENT_ID_COL, config.TARGET_COL]
    ]
    merged = scores.merge(feats, left_on="patient_id",
                          right_on=config.PATIENT_ID_COL, how="inner")
    return merged["probability"].to_numpy(), merged[config.TARGET_COL].to_numpy()


def sweep_thresholds(proba, y_true, steps: int = 101) -> pd.DataFrame:
    """Confusion counts + precision/recall/F1 across thresholds 0..1."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    rows = []
    for t in np.linspace(0, 1, steps):
        pred = proba >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
        flagged = tp + fp
        rows.append({
            "threshold": round(float(t), 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4) if prec == prec else None,
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "flagged": flagged,
            "flagged_pct": round(flagged / n * 100, 2),
        })
    return pd.DataFrame(rows)


def recommended_threshold(proba, y_true,
                          cost_fn: float = DEFAULT_COST_FN,
                          cost_fp: float = DEFAULT_COST_FP,
                          sweep: pd.DataFrame | None = None):
    """Threshold minimizing cost_fn*FN + cost_fp*FP. Returns (threshold, row, sweep)."""
    df = sweep if sweep is not None else sweep_thresholds(proba, y_true)
    df = df.copy()
    df["total_cost"] = df["fn"] * cost_fn + df["fp"] * cost_fp
    best = df.loc[df["total_cost"].idxmin()]
    return float(best["threshold"]), best, df


def run(verbose: bool = True) -> pd.DataFrame:
    proba, y = load_oof()
    sweep = sweep_thresholds(proba, y)
    out = config.OUTPUTS_DIR / "threshold_sweep.csv"
    sweep.to_csv(out, index=False)

    thr, best, _ = recommended_threshold(proba, y, sweep=sweep)
    if verbose:
        default = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]
        print(f"Sweep written -> {out}")
        print(f"\nDefault threshold 0.50: recall={default['recall']:.2f} "
              f"precision={default['precision']:.2f} caught={default['tp']:,} "
              f"missed={default['fn']:,} false_alarms={default['fp']:,}")
        print(f"Cost-optimal ({DEFAULT_COST_FN:.0f}:1) threshold {thr:.2f}: "
              f"recall={best['recall']:.2f} precision={best['precision']:.2f} "
              f"caught={best['tp']:,} missed={best['fn']:,} false_alarms={best['fp']:,}")
    return sweep


if __name__ == "__main__":
    run()
