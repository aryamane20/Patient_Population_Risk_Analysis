"""Unit tests for src/clean.py - schema + null handling + leakage guards."""

import numpy as np
import pandas as pd
import pytest

from src import clean, config


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """A tiny raw-shaped frame exercising each cleaning rule."""
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4, 5],
        "patient_nbr": [10, 10, 20, 30, 40],       # patient 10 duplicated
        "age": ["[70-80)", "[60-70)", "[80-90)", "[40-50)", "[50-60)"],
        "weight": ["?", "?", "?", "?", "?"],       # near-empty -> dropped
        "payer_code": ["MC", "?", "MC", "?", "MC"],
        "discharge_disposition_id": [1, 1, 11, 1, 1],  # id 11 = expired -> excluded
        "number_inpatient": [0, 1, 2, 3, 100],     # 100 = outlier
        "number_emergency": [0, 0, 0, 0, 0],
        "number_outpatient": [0, 0, 0, 0, 0],
        "time_in_hospital": [2, 3, 4, 5, 500],
        "num_medications": [5, 6, 7, 8, 9],
        "change": ["Ch", "No", "Ch", "No", "Ch"],
        "diabetesMed": ["Yes", "No", "Yes", "No", "Yes"],
        "readmitted": ["<30", ">30", "NO", "<30", "NO"],
    })


def test_replace_missing_token(raw_sample):
    out = clean.replace_missing_token(raw_sample)
    assert out["payer_code"].isna().sum() == 2
    assert out["weight"].isna().all()


def test_drop_sparse_columns(raw_sample):
    out = clean.drop_sparse_columns(raw_sample)
    for col in config.DROP_COLUMNS:
        assert col not in out.columns


def test_exclude_expired_hospice(raw_sample):
    out = clean.exclude_expired_hospice(raw_sample)
    assert 11 not in out["discharge_disposition_id"].values
    assert len(out) == len(raw_sample) - 1


def test_deduplicate_keeps_first_encounter(raw_sample):
    out = clean.deduplicate_patients(raw_sample)
    assert out["patient_nbr"].is_unique
    # patient 10 keeps its first encounter (encounter_id == 1)
    assert out.loc[out["patient_nbr"] == 10, "encounter_id"].iloc[0] == 1


def test_cap_outliers_clips_upper_tail(raw_sample):
    out = clean.cap_outliers(raw_sample, quantile=0.99)
    assert out["number_inpatient"].max() < 100
    assert out["time_in_hospital"].max() < 500


def test_build_target_is_binary_30d(raw_sample):
    out = clean.build_target(raw_sample)
    assert set(out[config.TARGET_COL].unique()) <= {0, 1}
    # exactly the two '<30' rows are positive
    assert out[config.TARGET_COL].sum() == 2


def test_standardize_age_midpoint(raw_sample):
    out = clean.standardize_fields(raw_sample)
    assert out.loc[0, "age_midpoint"] == 75.0
    assert out["change_flag"].isin([0, 1]).all()
    assert out["diabetes_med_flag"].isin([0, 1]).all()


def test_full_clean_pipeline_invariants(raw_sample):
    out, stats = clean.clean(raw_sample, verbose=False)
    assert out["patient_nbr"].is_unique                 # dedup held
    assert config.TARGET_COL in out.columns             # target built
    assert stats["rows_final"] == len(out)
    assert 0.0 <= stats["positive_rate"] <= 1.0
