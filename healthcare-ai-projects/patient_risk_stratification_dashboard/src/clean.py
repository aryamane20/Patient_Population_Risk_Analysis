"""
clean.py - turn the raw Diabetes-130 table into a clean, analysis-ready frame.

Each step is a small pure function so it can be unit-tested and reordered.
`clean()` runs them in the right order and returns the cleaned DataFrame plus a
dict of provenance stats (rows dropped at each stage) for the notebook/README.

Cleaning decisions (all documented in the plan / notebook):
  1. '?' -> NaN                        (single missing-value policy)
  2. drop near-empty columns           (weight ~97% missing, payer_code)
  3. exclude expired/hospice discharges (cannot be readmitted -> target leakage)
  4. de-duplicate to first encounter per patient (avoid within-patient leakage)
  5. cap extreme upper-tail outliers    (99th percentile)
  6. standardize: age bucket -> midpoint, Yes/No flags -> 0/1
  7. build the binary target readmit_30d
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ── age bucket "[70-80)" -> ordinal midpoint 75 ──
def _age_bucket_to_midpoint(value: str):
    if not isinstance(value, str) or "-" not in value:
        return np.nan
    lo, hi = value.strip("[]()").split("-")
    return (int(lo) + int(hi)) / 2.0


def replace_missing_token(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1 - '?' (and empty strings) -> NaN."""
    return df.replace({config.MISSING_TOKEN: np.nan, "": np.nan})


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 - drop columns that are near-empty or leakage-prone."""
    cols = [c for c in config.DROP_COLUMNS if c in df.columns]
    return df.drop(columns=cols)


def exclude_expired_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 - remove encounters the patient could not survive to be readmitted from."""
    if "discharge_disposition_id" not in df.columns:
        return df
    mask = ~df["discharge_disposition_id"].isin(config.EXPIRED_HOSPICE_DISPOSITIONS)
    return df[mask].copy()


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 - keep the first encounter per patient (chronological by encounter_id)."""
    if config.PATIENT_ID_COL not in df.columns:
        return df
    ordered = df.sort_values(config.ENCOUNTER_ID_COL)
    return ordered.drop_duplicates(subset=config.PATIENT_ID_COL, keep="first").copy()


def cap_outliers(df: pd.DataFrame, quantile: float = config.OUTLIER_UPPER_QUANTILE) -> pd.DataFrame:
    """Step 5 - cap the upper tail of skewed count columns at the given quantile."""
    df = df.copy()
    for col in config.OUTLIER_CAP_COLS:
        if col in df.columns:
            cap = df[col].quantile(quantile)
            df[col] = df[col].clip(upper=cap)
    return df


def standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6 - age bucket -> midpoint; change/diabetesMed -> 0/1."""
    df = df.copy()
    if "age" in df.columns:
        df["age_midpoint"] = df["age"].map(_age_bucket_to_midpoint)
    if "change" in df.columns:
        df["change_flag"] = (df["change"] == "Ch").astype(int)
    if "diabetesMed" in df.columns:
        df["diabetes_med_flag"] = (df["diabetesMed"] == "Yes").astype(int)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Step 7 - binary target: 1 if readmitted within 30 days, else 0."""
    df = df.copy()
    df[config.TARGET_COL] = (df[config.RAW_TARGET_COL] == "<30").astype(int)
    return df


def clean(df: pd.DataFrame, verbose: bool = True):
    """Run all cleaning steps; return (clean_df, stats)."""
    stats = {"rows_raw": len(df)}

    df = replace_missing_token(df)
    df = drop_sparse_columns(df)

    df = exclude_expired_hospice(df)
    stats["rows_after_exclusions"] = len(df)

    df = deduplicate_patients(df)
    stats["rows_after_dedup"] = len(df)

    df = cap_outliers(df)
    df = standardize_fields(df)
    df = build_target(df)

    stats["positive_rate"] = round(df[config.TARGET_COL].mean(), 4)
    stats["rows_final"] = len(df)

    if verbose:
        print("Cleaning provenance:")
        print(f"  raw rows                : {stats['rows_raw']:,}")
        print(f"  after expired/hospice   : {stats['rows_after_exclusions']:,} "
              f"(-{stats['rows_raw'] - stats['rows_after_exclusions']:,})")
        print(f"  after patient dedup     : {stats['rows_after_dedup']:,} "
              f"(-{stats['rows_after_exclusions'] - stats['rows_after_dedup']:,})")
        print(f"  final rows              : {stats['rows_final']:,}")
        print(f"  positive rate (readmit) : {stats['positive_rate']*100:.2f}%")

    return df, stats


if __name__ == "__main__":
    from .ingest import load_raw
    raw = load_raw()
    clean_df, stats = clean(raw)
    # CSV (not Parquet) per plan decision #4 - avoids a pyarrow dependency.
    out = config.DATA_PROCESSED / "cleaned.csv"
    clean_df.to_csv(out, index=False)
    print(f"\nSaved cleaned data -> {out}")
