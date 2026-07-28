"""
Healthcare FHIR Data Pipeline — Main / Entry Point
Mindbowser Learning Project — Day 14

This file only CALLS the functions defined in pipeline_functions.py.
It orchestrates the run, applies the rules-based risk stratification,
and exports the final de-identified CSV (with risk columns included).

Run with:  python3 main.py
"""

import glob
import os
import re
from datetime import datetime

import pandas as pd

from pipeline_functions import run_pipeline, calculate_simple_risk_score


def next_version_number():
    """
    Find the next report version number by scanning existing
    patient_features_v###.csv files and returning max + 1.
    Starts at 1 when no prior reports exist.
    """
    existing = glob.glob("patient_features_v*.csv")
    versions = []
    for path in existing:
        match = re.search(r"patient_features_v(\d+)\.csv$", os.path.basename(path))
        if match:
            versions.append(int(match.group(1)))
    return (max(versions) + 1) if versions else 1


def export_dataset(df):
    """
    Write the de-identified feature dataset (with risk columns) to CSV.

    Two files are written each run:
      1. An incrementing version snapshot, patient_features_v###.csv
         (v001, v002, ...) — a permanent, uniquely-named record of this
         run. The highest number is always the newest. Nothing is ever
         overwritten, so every past version is preserved.
      2. patient_features_LATEST.csv — a convenience copy of this newest
         run, so you can open the current report at a glance without
         hunting for the highest version number.
    """
    version = next_version_number()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    output_filename = f"patient_features_v{version:03d}.csv"
    latest_filename = "patient_features_LATEST.csv"

    df.to_csv(output_filename, index=False)
    df.to_csv(latest_filename, index=False)

    print(f"\n✅ De-identified feature dataset exported ({run_time}):")
    print(f"   Newest (always current): {latest_filename}")
    print(f"   This run's snapshot:      {output_filename}  (version {version})")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   PHI removed: names, DOB, phone, ZIP, real patient ID")
    print(f"   Safe Harbor compliant: Yes")
    print(f"\nThis file is ready for ML model training.")
    print(f"No BAA required for this de-identified dataset.")
    return output_filename


def show_sample(df):
    """Print the first few rows of a few illustrative columns."""
    print("\n--- SAMPLE OUTPUT (first 3 rows, selected columns) ---")
    display_cols = [
        "patient_token", "age", "gender_female",
        "num_active_conditions", "has_diabetes",
        "hba1c_value", "num_inpatient_encounters",
        "high_utilizer", "high_comorbidity_burden"
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(3).to_string(index=False))


def add_risk_stratification(df):
    """
    Apply the rules-based risk score and tier to the DataFrame.
    (Rules-based demo, not a trained model.)
    """
    df["risk_score"] = df.apply(calculate_simple_risk_score, axis=1)
    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[0, 30, 60, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True  # so a score of exactly 0 maps to LOW, not NaN
    )
    return df


def show_risk_summary(df):
    """Print the risk-tier distribution and the highest-risk patients."""
    print("\n--- RISK TIER DISTRIBUTION ---")
    print(df["risk_tier"].value_counts())
    print("\nTop 3 highest risk patients:")
    print(df.nlargest(3, "risk_score")[
        ["patient_token", "risk_score", "risk_tier",
         "num_inpatient_encounters", "has_heart_failure",
         "num_abnormal_labs"]
    ].to_string(index=False))


if __name__ == "__main__":
    # Run pipeline with 15 patients
    # Increase num_patients for more data
    # (each patient requires ~4-5 API calls so be mindful of rate limits)
    result_df = run_pipeline(num_patients=15)

    if result_df is not None:
        # ── Risk stratification (applied before export so it persists) ──
        result_df = add_risk_stratification(result_df)

        # ── Export the de-identified dataset (now includes risk columns) ──
        export_dataset(result_df)

        # ── Reporting ──
        show_sample(result_df)
        show_risk_summary(result_df)

        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset shape: {result_df.shape}")
        print(f"Memory usage: {result_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        print(f"\nNext step: Load this CSV into your ML training notebook")
        print(f"and train a readmission prediction model.")
