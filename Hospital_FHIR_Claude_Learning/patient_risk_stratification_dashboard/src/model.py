"""
model.py - preprocessing + models + evaluation (Steps 4–5).

Key production/rigor choices:
- ONE sklearn Pipeline = ColumnTransformer(preprocess) + estimator. Preprocessing
  is fit on TRAIN ONLY (inside cross-validation / the split), so there is no
  leakage, and the fitted preprocessing travels with the serialized model.
- Two models: LogisticRegression (interpretable baseline) and a gradient-boosted
  champion (XGBoost, or HistGradientBoosting as a dependency-free fallback), both
  made imbalance-aware for the ~9% positive rate.
- Champion chosen by PR-AUC (the imbalance-aware metric), not raw accuracy.

Reusable helpers (`get_feature_data`, `make_pipeline`) are imported by export.py
so the full-population out-of-fold scoring uses the identical model definition.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import FEATURE_SCHEMA, FeatureSchema, build_features

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - fallback path
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_XGB = False


# ─────────────────────────────────────────────
# Data access
# ─────────────────────────────────────────────
def get_feature_data():
    """
    Load engineered features. Returns (X, y, ids, schema).

    Rebuilds from cleaned.csv (or from scratch) if features.csv is absent, so the
    module is runnable standalone.
    """
    feat_path = config.DATA_PROCESSED / "features.csv"
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        schema = FEATURE_SCHEMA
    else:
        clean_path = config.DATA_PROCESSED / "cleaned.csv"
        if clean_path.exists():
            clean_df = pd.read_csv(clean_path)
        else:
            from .clean import clean
            from .ingest import load_raw
            clean_df, _ = clean(load_raw())
        df, schema = build_features(clean_df, verbose=False)

    X = df[schema.all_features].copy()
    y = df[config.TARGET_COL].astype(int).to_numpy()
    ids = df[config.PATIENT_ID_COL].copy()
    return X, y, ids, schema


# ─────────────────────────────────────────────
# Preprocessing + estimators
# ─────────────────────────────────────────────
def make_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, schema.numeric),
        ("cat", categorical, schema.categorical),
    ])


def _make_estimator(kind: str, scale_pos_weight: float):
    if kind == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_SEED,
        )
    if kind == "gbm":
        if _HAS_XGB:
            return XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            )
        return HistGradientBoostingClassifier(  # fallback
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
    raise ValueError(f"unknown model kind: {kind!r}")


def make_pipeline(kind: str, schema: FeatureSchema, scale_pos_weight: float = 1.0) -> Pipeline:
    """Full preprocess + estimator pipeline for `kind` in {'logreg','gbm'}."""
    return Pipeline([
        ("preprocess", make_preprocessor(schema)),
        ("model", _make_estimator(kind, scale_pos_weight)),
    ])


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return round(neg / pos, 4) if pos else 1.0


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
             threshold: float = 0.5) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "auroc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ─────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────
def run(verbose: bool = True) -> dict:
    X, y, ids, schema = get_feature_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    spw = compute_scale_pos_weight(y_train)

    results = {
        "n_rows": int(len(X)),
        "n_features": len(schema.all_features),
        "positive_rate": round(float(np.mean(y)), 4),
        "scale_pos_weight": spw,
        "gbm_backend": "xgboost" if _HAS_XGB else "hist_gradient_boosting",
        "models": {},
    }

    fitted = {}
    for kind in ("logreg", "gbm"):
        pipe = make_pipeline(kind, schema, scale_pos_weight=spw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results["models"][kind] = metrics
        fitted[kind] = pipe
        if verbose:
            print(f"[{kind}] AUROC={metrics['auroc']}  PR-AUC={metrics['pr_auc']}  "
                  f"recall={metrics['recall']}  precision={metrics['precision']}")

    # Champion by PR-AUC (imbalance-aware).
    champion = max(results["models"], key=lambda k: results["models"][k]["pr_auc"])
    results["champion"] = champion

    champ_path = config.MODELS_DIR / "champion_pipeline.joblib"
    joblib.dump(fitted[champion], champ_path)
    joblib.dump(fitted["logreg"], config.MODELS_DIR / "logreg_pipeline.joblib")

    metrics_path = config.MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    if verbose:
        print(f"\nChampion: {champion}  ->  {champ_path.name}")
        print(f"Metrics written -> {metrics_path}")
        # Literature sanity check for this dataset (de-duplicated).
        gbm_auroc = results["models"]["gbm"]["auroc"]
        if not (0.60 <= gbm_auroc <= 0.72):
            print(f"  NOTE: GBM AUROC={gbm_auroc} is outside the ~0.64–0.70 "
                  f"literature range - inspect for leakage or a bug.")

    return results


if __name__ == "__main__":
    run()
