"""
risk_scoring.py - turn model probabilities into an actionable 0–100 score + tiers.

Why quantile tiers (plan decision #3): this dataset's readmission probabilities
cluster low, so fixed probability cutoffs (e.g. >0.5 = High) would leave the High
list empty and useless. Quantiles guarantee a workable-sized High list for care
coordinators regardless of the model's absolute calibration:

    High   = top 20% of scores (>= P80)
    Medium = P50 – P80
    Low    = < P50

This mirrors the pd.cut / np.select tiering pattern from
population_analytics/population_health.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def percent_of_total(counts, decimals: int = 1) -> np.ndarray:
    """
    Percentages that ALWAYS sum to exactly 100 (largest-remainder method).

    Plain rounding (incl. pandas' round-half-to-even) can make displayed
    percentages sum to 99.9 or 100.1. This distributes the rounding residual to
    the largest fractional parts so the total is exactly 100.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts)
    scale = 10 ** decimals
    raw = counts / total * 100 * scale
    floored = np.floor(raw)
    remainder = int(round(scale * 100 - floored.sum()))
    order = np.argsort(-(raw - floored))  # largest fractional parts first
    for i in range(remainder):
        floored[order[i % len(floored)]] += 1
    return floored / scale


def to_risk_score(proba) -> np.ndarray:
    """P(readmit ≤30d) in [0,1] -> integer 0–100 risk score."""
    proba = np.asarray(proba, dtype=float)
    return np.clip(np.round(proba * 100), 0, 100).astype(int)


def tier_thresholds(scores) -> dict:
    """Compute the P50 / P80 cut points used to assign tiers."""
    scores = np.asarray(scores, dtype=float)
    # TIER_QUANTILES = [0.0, 0.50, 0.80, 1.0]; the interior cuts are P50 and P80.
    p_med, p_high = config.TIER_QUANTILES[1], config.TIER_QUANTILES[2]
    return {
        "medium_cut": float(np.quantile(scores, p_med)),
        "high_cut": float(np.quantile(scores, p_high)),
    }


def assign_tiers(scores, thresholds: dict | None = None) -> np.ndarray:
    """Assign Low/Medium/High by quantile thresholds (ties break upward)."""
    scores = np.asarray(scores, dtype=float)
    th = thresholds or tier_thresholds(scores)
    conditions = [
        scores >= th["high_cut"],
        scores >= th["medium_cut"],
    ]
    choices = ["High", "Medium"]
    return np.select(conditions, choices, default="Low")


def score_frame(ids, proba, id_col: str = config.PATIENT_ID_COL) -> pd.DataFrame:
    """Build a tidy frame: id, probability, risk_score, tier."""
    proba = np.asarray(proba, dtype=float)
    scores = to_risk_score(proba)
    th = tier_thresholds(scores)
    tiers = assign_tiers(scores, th)
    return pd.DataFrame({
        id_col: np.asarray(ids),
        "probability": np.round(proba, 4),
        "risk_score": scores,
        "tier": tiers,
    })


def tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count + mean score per tier, ordered High -> Low, for the dashboard."""
    order = ["High", "Medium", "Low"]
    g = (df.groupby("tier")
           .agg(patients=("risk_score", "size"),
                mean_score=("risk_score", "mean"),
                mean_probability=("probability", "mean"))
           .reindex(order)
           .dropna(how="all"))
    g["mean_score"] = g["mean_score"].round(1)
    g["mean_probability"] = g["mean_probability"].round(4)
    g["pct_of_panel"] = percent_of_total(g["patients"])
    return g.reset_index()


if __name__ == "__main__":
    rng = np.random.default_rng(config.RANDOM_SEED)
    demo = rng.beta(2, 20, size=1000)  # low-clustered like the real scores
    frame = score_frame(np.arange(1000), demo)
    print("Tier thresholds:", tier_thresholds(frame["risk_score"]))
    print(tier_summary(frame).to_string(index=False))
