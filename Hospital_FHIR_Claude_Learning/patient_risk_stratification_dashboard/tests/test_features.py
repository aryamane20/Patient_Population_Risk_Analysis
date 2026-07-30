"""Unit tests for src/features.py - ICD-9 grouping, schema, engineered columns."""

import numpy as np
import pandas as pd
import pytest

from src import config, features


@pytest.mark.parametrize("code,expected", [
    ("250.7", "Diabetes"),
    ("250", "Diabetes"),
    ("428", "Circulatory"),   # heart failure
    ("785", "Circulatory"),
    ("491", "Respiratory"),
    ("786", "Respiratory"),
    ("550", "Digestive"),
    ("820", "Injury"),
    ("715", "Musculoskeletal"),
    ("600", "Genitourinary"),
    ("788", "Genitourinary"),
    ("199", "Neoplasm"),
    ("V57", "Other"),
    ("E909", "Injury"),
    ("somegarbage", "Other"),
    (np.nan, "Missing"),
])
def test_icd9_to_group(code, expected):
    assert features.icd9_to_group(code) == expected


@pytest.fixture
def clean_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_nbr": [1, 2, 3],
        "encounter_id": [11, 22, 33],
        "readmit_30d": [1, 0, 0],
        "age_midpoint": [75.0, 65.0, 55.0],
        "time_in_hospital": [3, 4, 5],
        "num_lab_procedures": [40, 50, 60],
        "num_procedures": [1, 2, 3],
        "num_medications": [10, 12, 14],
        "number_outpatient": [0, 1, 2],
        "number_emergency": [1, 0, 3],
        "number_inpatient": [2, 0, 1],
        "number_diagnoses": [9, 7, 5],
        "admission_type_id": [1, 3, 6],
        "admission_source_id": [7, 1, 17],
        "discharge_disposition_id": [1, 6, 2],
        "diag_1": ["428", "250.7", "V57"],
        "diag_2": ["276", "428", "250"],
        "diag_3": ["414", "?", "820"],
        "A1Cresult": [">8", np.nan, "Norm"],
        "max_glu_serum": [np.nan, ">200", "Norm"],
        "change": ["Ch", "No", "Ch"],
        "diabetesMed": ["Yes", "Yes", "No"],
        "change_flag": [1, 0, 1],
        "diabetes_med_flag": [1, 1, 0],
        "race": ["Caucasian", np.nan, "AfricanAmerican"],
        "gender": ["Male", "Female", "Male"],
        "medical_specialty": ["Cardiology", np.nan, "Surgery"],
        "insulin": ["Up", "No", "Steady"],
        # a few medication columns for the churn counter
        "metformin": ["Steady", "No", "Up"],
        "glipizide": ["No", "Down", "No"],
    })


def test_utilization_features(clean_sample):
    out = features.add_utilization_features(clean_sample)
    # num_prior_admissions = inpatient + emergency + outpatient
    assert out.loc[0, "num_prior_admissions"] == 3  # 2 + 1 + 0
    assert out.loc[2, "num_prior_admissions"] == 6  # 1 + 3 + 2
    assert (out["num_chronic_conditions"] == clean_sample["number_diagnoses"]).all()


def test_medication_features_count_changes(clean_sample):
    out = features.add_medication_features(clean_sample)
    # insulin is one of the 23 medication columns, so it counts too.
    # row 0: insulin Up + metformin Steady + glipizide No -> 1 changed, 2 on
    assert out.loc[0, "num_meds_changed"] == 1
    assert out.loc[0, "num_meds_on"] == 2
    # row 2: insulin Steady + metformin Up + glipizide No -> 1 changed, 2 on
    assert out.loc[2, "num_meds_changed"] == 1
    assert out.loc[2, "num_meds_on"] == 2


def test_lab_features_fill_none(clean_sample):
    out = features.add_lab_features(clean_sample)
    assert out.loc[1, "a1c_result"] == "None"
    assert out.loc[0, "max_glu_serum_grp"] == "None"


def test_build_features_schema_and_no_missing_columns(clean_sample):
    feat_df, schema = features.build_features(clean_sample, verbose=False)
    for col in schema.all_features:
        assert col in feat_df.columns, f"missing engineered column {col}"
    assert config.TARGET_COL in feat_df.columns
    assert config.PATIENT_ID_COL in feat_df.columns
    assert len(feat_df) == len(clean_sample)


def test_diagnosis_groups_applied(clean_sample):
    out = features.add_diagnosis_groups(clean_sample)
    assert out.loc[0, "diag_1_group"] == "Circulatory"  # 428
    assert out.loc[1, "diag_1_group"] == "Diabetes"     # 250.7
