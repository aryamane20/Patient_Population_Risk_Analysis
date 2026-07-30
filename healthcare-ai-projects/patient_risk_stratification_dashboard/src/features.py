"""
features.py - turn the cleaned table into model-ready features + the feature
"contract" (schema) that both CSVSource and the production FHIRSource must honor.

Design intent:
- Every engineered feature is built by a small, testable function.
- `build_features()` runs them and returns (df, schema) where `schema` lists the
  NUMERIC and CATEGORICAL feature columns. `model.py` consumes that schema to
  build its ColumnTransformer, so the columns are defined in exactly one place.
- The ICD-9 -> disease-group mapping follows Strack et al. 2014 (the dataset's
  origin paper), which is the standard, citable grouping for this data.

Nothing here fits any statistic on the data (no scaling/imputation), so it is
safe to run on the full frame before the train/test split - the leakage-prone
steps (impute/scale/one-hot) live in model.py and are fit on train only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


# ─────────────────────────────────────────────
# Feature schema (the "contract")
# ─────────────────────────────────────────────
@dataclass
class FeatureSchema:
    """The columns the model consumes. Shared by every data source."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def all_features(self) -> list[str]:
        return self.numeric + self.categorical


NUMERIC_FEATURES = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "num_chronic_conditions",   # proxy = number_diagnoses
    "num_prior_admissions",     # inpatient + emergency + outpatient (prior year)
    "num_meds_changed",         # count of the 23 drugs adjusted up/down
    "num_meds_on",              # count of the 23 drugs actively prescribed
    "change_flag",              # any medication change (0/1)
    "diabetes_med_flag",        # any diabetes medication (0/1)
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "medical_specialty_grp",
    "admission_type_grp",
    "admission_source_grp",
    "discharge_disposition_grp",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "a1c_result",
    "max_glu_serum_grp",
    "insulin",
]

FEATURE_SCHEMA = FeatureSchema(
    numeric=list(NUMERIC_FEATURES),
    categorical=list(CATEGORICAL_FEATURES),
)


# ─────────────────────────────────────────────
# ICD-9 diagnosis grouping (Strack et al. 2014)
# ─────────────────────────────────────────────
def icd9_to_group(code) -> str:
    """Map a raw ICD-9 diagnosis code (e.g. '250.7', '428', 'V57') to a group."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s == "" or s.lower() == "nan":
        return "Missing"
    # V-codes (supplementary) and E-codes (external injury cause) -> Other/Injury.
    if s.startswith("V"):
        return "Other"
    if s.startswith("E"):
        return "Injury"
    try:
        num = float(s)
    except ValueError:
        return "Other"
    icd = int(num)
    # Diabetes is coded 250.xx and is called out explicitly in Strack 2014.
    if 250 <= num < 251:
        return "Diabetes"
    if (390 <= icd <= 459) or icd == 785:
        return "Circulatory"
    if (460 <= icd <= 519) or icd == 786:
        return "Respiratory"
    if (520 <= icd <= 579) or icd == 787:
        return "Digestive"
    if 800 <= icd <= 999:
        return "Injury"
    if 710 <= icd <= 739:
        return "Musculoskeletal"
    if (580 <= icd <= 629) or icd == 788:
        return "Genitourinary"
    if 140 <= icd <= 239:
        return "Neoplasm"
    return "Other"


# ─────────────────────────────────────────────
# ID -> human-readable group mappings
# (simplified from data/raw/IDS_mapping.csv; unknown/null codes collapse cleanly)
# ─────────────────────────────────────────────
_ADMISSION_TYPE_GRP = {
    1: "Emergency", 2: "Urgent", 3: "Elective",
    4: "Newborn", 7: "Trauma",
    5: "Unknown", 6: "Unknown", 8: "Unknown",
}
_ADMISSION_SOURCE_GRP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer",
    18: "Transfer", 22: "Transfer", 25: "Transfer",
    7: "Emergency Room",
    11: "Birth", 12: "Birth", 13: "Birth", 14: "Birth",
}
_DISCHARGE_GRP = {
    1: "Home",
    6: "Home Health", 8: "Home Health",
    2: "Transferred Facility", 3: "Transferred Facility",
    4: "Transferred Facility", 5: "Transferred Facility",
    22: "Transferred Facility", 23: "Transferred Facility",
    24: "Transferred Facility",
    7: "AMA",  # left against medical advice
}


def _map_grouped(series: pd.Series, mapping: dict, default: str = "Other") -> pd.Series:
    return series.map(lambda v: mapping.get(v, default) if pd.notna(v) else "Unknown")


# ─────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────
def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prior-year utilization: the strongest readmission signal in this dataset."""
    df = df.copy()
    df["num_prior_admissions"] = (
        df["number_inpatient"].fillna(0)
        + df["number_emergency"].fillna(0)
        + df["number_outpatient"].fillna(0)
    )
    df["num_chronic_conditions"] = df["number_diagnoses"]
    return df


def add_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Map diag_1/2/3 ICD-9 codes to disease groups (Strack 2014)."""
    df = df.copy()
    for raw, out in (("diag_1", "diag_1_group"),
                     ("diag_2", "diag_2_group"),
                     ("diag_3", "diag_3_group")):
        df[out] = df[raw].map(icd9_to_group) if raw in df.columns else "Missing"
    return df


def add_grouped_admin_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket the high-cardinality admin ID codes into interpretable groups."""
    df = df.copy()
    if "admission_type_id" in df.columns:
        df["admission_type_grp"] = _map_grouped(df["admission_type_id"], _ADMISSION_TYPE_GRP)
    if "admission_source_id" in df.columns:
        df["admission_source_grp"] = _map_grouped(df["admission_source_id"], _ADMISSION_SOURCE_GRP)
    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_grp"] = _map_grouped(df["discharge_disposition_id"], _DISCHARGE_GRP)
    return df


def add_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize lab result columns; missing tests become an explicit 'None' level."""
    df = df.copy()
    if "A1Cresult" in df.columns:
        df["a1c_result"] = df["A1Cresult"].fillna("None")
    if "max_glu_serum" in df.columns:
        df["max_glu_serum_grp"] = df["max_glu_serum"].fillna("None")
    return df


def add_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """Medication churn/burden across the 23 drug columns."""
    df = df.copy()
    med_cols = [c for c in config.MEDICATION_COLUMNS if c in df.columns]
    changed = np.zeros(len(df), dtype=int)
    on = np.zeros(len(df), dtype=int)
    for c in med_cols:
        col = df[c].fillna("No")
        changed += col.isin(["Up", "Down"]).to_numpy(dtype=int)
        on += (col != "No").to_numpy(dtype=int)
    df["num_meds_changed"] = changed
    df["num_meds_on"] = on
    return df


def collapse_rare_categories(df: pd.DataFrame,
                             columns: list[str],
                             min_frac: float = config.RARE_CATEGORY_MIN_FRAC) -> pd.DataFrame:
    """Collapse categories rarer than `min_frac` into 'Other' (e.g. medical_specialty)."""
    df = df.copy()
    n = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(dropna=False, normalize=True)
        rare = set(freq[freq < min_frac].index)
        df[col] = df[col].where(~df[col].isin(rare), "Other")
    return df


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill categorical NaNs with an explicit 'Unknown' level so they are modeled."""
    df = df.copy()
    for col in ("race", "gender", "insulin"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").replace("", "Unknown")
    return df


def build_features(df: pd.DataFrame, verbose: bool = True):
    """
    Run all feature builders. Returns (feature_df, schema).

    `feature_df` keeps the id + target columns alongside the engineered features
    so downstream code (split, scoring, export) has everything it needs.
    """
    df = add_utilization_features(df)
    df = add_diagnosis_groups(df)
    df = add_grouped_admin_fields(df)
    df = add_lab_features(df)
    df = add_medication_features(df)

    # bucket the one genuinely high-cardinality free-text categorical
    df = collapse_rare_categories(df, ["medical_specialty"])
    if "medical_specialty" in df.columns:
        df["medical_specialty_grp"] = df["medical_specialty"].fillna("Unknown").replace("", "Unknown")

    df = normalize_categoricals(df)

    keep = (
        [config.PATIENT_ID_COL, config.ENCOUNTER_ID_COL, config.TARGET_COL]
        + FEATURE_SCHEMA.all_features
    )
    keep = [c for c in keep if c in df.columns]
    feature_df = df[keep].copy()

    if verbose:
        print("Feature engineering:")
        print(f"  numeric features     : {len(FEATURE_SCHEMA.numeric)}")
        print(f"  categorical features : {len(FEATURE_SCHEMA.categorical)}")
        print(f"  total feature columns: {len(FEATURE_SCHEMA.all_features)}")
        print(f"  output shape         : {feature_df.shape}")
        missing = [c for c in FEATURE_SCHEMA.all_features if c not in feature_df.columns]
        if missing:
            print(f"  WARNING missing cols : {missing}")

    return feature_df, FEATURE_SCHEMA


if __name__ == "__main__":
    import pandas as pd

    from . import config

    src = config.DATA_PROCESSED / "cleaned.csv"
    if not src.exists():
        from .ingest import load_raw
        from .clean import clean
        raw = load_raw()
        clean_df, _ = clean(raw)
    else:
        clean_df = pd.read_csv(src)

    feats, schema = build_features(clean_df)
    out = config.DATA_PROCESSED / "features.csv"
    feats.to_csv(out, index=False)
    print(f"\nSaved features -> {out}")
    print("\nSample of engineered columns:")
    preview = [c for c in ("num_prior_admissions", "diag_1_group",
                           "admission_type_grp", "a1c_result",
                           "num_meds_changed") if c in feats.columns]
    print(feats[preview].head())
    print("\nDiagnosis-group distribution (diag_1):")
    print(feats["diag_1_group"].value_counts())
