"""
explain.py - SHAP explainability (Step 7).

Two deliverables:
  1. GLOBAL: which features drive readmission risk across the whole panel
     (summary bar plot saved to outputs/).
  2. PER-PATIENT: the 3 features with the largest *positive* SHAP contribution for
     each patient, rendered as human-readable phrases (e.g. "3 prior inpatient
     visits", "primary diagnosis: Circulatory"). This is the `top_3_risk_factors`
     column care coordinators actually act on.

We run SHAP on the champion's gradient-boosted tree via shap.TreeExplainer, using
the *transformed* matrix from the pipeline's ColumnTransformer so SHAP sees the
exact inputs the model saw.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse

from . import config
from .features import FEATURE_SCHEMA


# human-readable stems for each engineered feature
FRIENDLY_NAMES = {
    "num_prior_admissions": "prior admissions",
    "number_inpatient": "prior inpatient visits",
    "number_emergency": "prior ER visits",
    "number_outpatient": "prior outpatient visits",
    "num_chronic_conditions": "diagnoses on record",
    "time_in_hospital": "days in hospital",
    "num_medications": "medications",
    "num_lab_procedures": "lab procedures",
    "num_procedures": "procedures",
    "age_midpoint": "age",
    "num_meds_changed": "medications changed",
    "num_meds_on": "active medications",
    "change_flag": "medication changed this stay",
    "diabetes_med_flag": "on diabetes medication",
    "diag_1_group": "primary diagnosis",
    "diag_2_group": "secondary diagnosis",
    "diag_3_group": "additional diagnosis",
    "admission_type_grp": "admission type",
    "admission_source_grp": "admission source",
    "discharge_disposition_grp": "discharge disposition",
    "a1c_result": "HbA1c",
    "max_glu_serum_grp": "glucose serum",
    "insulin": "insulin",
    "race": "race",
    "gender": "gender",
    "medical_specialty_grp": "specialty",
}

_NUMERIC_SET = set(FEATURE_SCHEMA.numeric)


def _split_transformed_name(name: str):
    """'num__age_midpoint' -> ('age_midpoint', None); 'cat__diag_1_group_Circulatory'
    -> ('diag_1_group', 'Circulatory')."""
    body = name.split("__", 1)[1] if "__" in name else name
    for base in FEATURE_SCHEMA.all_features:
        if body == base:
            return base, None
        if body.startswith(base + "_"):
            return base, body[len(base) + 1:]
    return body, None


def _readable_factor(base: str, orig_row: pd.Series) -> str:
    """Human phrase for a BASE feature using the patient's ACTUAL value."""
    label = FRIENDLY_NAMES.get(base, base.replace("_", " "))
    val = orig_row.get(base, None)
    if pd.isna(val):
        return label
    if base in _NUMERIC_SET:
        num = int(val) if float(val).is_integer() else round(float(val), 1)
        return f"{label}: {num}"
    return f"{label}: {val}"


def get_shap_values(pipe, X: pd.DataFrame):
    """Return (shap_matrix [n,m], feature_names [m])."""
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_trans = preprocess.transform(X)
    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()
    feature_names = list(preprocess.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):          # some backends return [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, m, classes)
        sv = sv[:, :, 1]
    return sv, feature_names


def global_importance_plot(shap_matrix, feature_names, X_display=None,
                           out_path=None, max_display: int = 15):
    out_path = out_path or (config.OUTPUTS_DIR / "shap_global_importance.png")
    plt.figure()
    shap.summary_plot(
        shap_matrix, features=X_display, feature_names=feature_names,
        plot_type="bar", max_display=max_display, show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    return out_path


def _aggregate_by_base(shap_matrix, feature_names):
    """
    Collapse one-hot columns back to their base feature by summing SHAP values.

    A patient has exactly one primary diagnosis, so the risk it contributes is the
    SUM of that feature's one-hot columns - not each category listed separately.
    Returns (base_shap [n, n_base], base_names).
    """
    bases = [_split_transformed_name(n)[0] for n in feature_names]
    base_names = list(dict.fromkeys(bases))  # unique, order-preserving
    idx = {b: i for i, b in enumerate(base_names)}
    out = np.zeros((shap_matrix.shape[0], len(base_names)), dtype=float)
    for col, b in enumerate(bases):
        out[:, idx[b]] += shap_matrix[:, col]
    return out, base_names


def per_patient_top_factors(shap_matrix, feature_names, X: pd.DataFrame,
                            k: int = 3) -> list[str]:
    """For each row, the k BASE features with the largest positive SHAP value,
    labeled with the patient's actual value."""
    base_shap, base_names = _aggregate_by_base(shap_matrix, feature_names)
    base_arr = np.array(base_names)
    results: list[str] = []
    X = X.reset_index(drop=True)
    for i in range(base_shap.shape[0]):
        row_sv = base_shap[i]
        order = np.argsort(row_sv)[::-1]
        picks = [j for j in order if row_sv[j] > 0][:k]
        phrases = [_readable_factor(base_arr[j], X.iloc[i]) for j in picks]
        results.append("; ".join(phrases) if phrases else "no elevated risk factors")
    return results


def global_importance_frame(shap_matrix, feature_names, top_n: int = 12) -> pd.DataFrame:
    """
    Plain-language global driver table for the dashboard.

    Aggregates one-hot columns back to base features, measures each feature's
    average absolute push on risk, and expresses it as a share of total model
    influence ("relative influence"). Returns [feature, label, relative_influence].
    """
    base_shap, base_names = _aggregate_by_base(shap_matrix, feature_names)
    importance = np.abs(base_shap).mean(axis=0)
    total = importance.sum() or 1.0
    df = pd.DataFrame({
        "feature": base_names,
        "label": [FRIENDLY_NAMES.get(b, b.replace("_", " ")) for b in base_names],
        "relative_influence": importance / total * 100,
    })
    return (df.sort_values("relative_influence", ascending=False)
              .head(top_n).reset_index(drop=True))


def patient_contributions(pipe, X_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    SHAP contributions for a SINGLE patient, aggregated to base features.

    Returns a DataFrame [feature, label, shap_value, patient_value] sorted by
    absolute contribution - ready to render as a horizontal bar in the dashboard
    drill-down (positive = pushes risk up, negative = pulls risk down).
    """
    if len(X_row) != 1:
        raise ValueError("patient_contributions expects exactly one row")
    shap_matrix, feature_names = get_shap_values(pipe, X_row)
    base_shap, base_names = _aggregate_by_base(shap_matrix, feature_names)
    row = base_shap[0]
    orig = X_row.iloc[0]
    out = pd.DataFrame({
        "feature": base_names,
        "label": [FRIENDLY_NAMES.get(b, b.replace("_", " ")) for b in base_names],
        "shap_value": row,
        "patient_value": [orig.get(b, None) for b in base_names],
    })
    out = out.reindex(out["shap_value"].abs().sort_values(ascending=False).index)
    return out.head(top_n).reset_index(drop=True)


def explain(pipe, X: pd.DataFrame, make_global_plot: bool = True, k: int = 3):
    """Convenience: returns (top_factors list, shap_matrix, feature_names)."""
    shap_matrix, feature_names = get_shap_values(pipe, X)
    if make_global_plot:
        path = global_importance_plot(shap_matrix, feature_names, X_display=None)
        print(f"Saved global SHAP importance -> {path}")
    top = per_patient_top_factors(shap_matrix, feature_names, X, k=k)
    return top, shap_matrix, feature_names


if __name__ == "__main__":
    import joblib

    from .model import get_feature_data

    pipe = joblib.load(config.MODELS_DIR / "champion_pipeline.joblib")
    X, y, ids, schema = get_feature_data()
    sample = X.head(500)
    top, sv, names = explain(pipe, sample, make_global_plot=True)
    print("\nExample per-patient top-3 risk factors:")
    for t in top[:5]:
        print("  -", t)
