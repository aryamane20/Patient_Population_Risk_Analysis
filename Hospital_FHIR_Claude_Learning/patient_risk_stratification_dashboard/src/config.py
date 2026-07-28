"""
config.py — single source of truth for paths, constants, and column groups.

Keeping these here (not scattered across modules) is what makes the pipeline
reproducible and easy to point at a different environment or data source.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths (all relative to the project root)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ─────────────────────────────────────────────
# Data source (UCI Diabetes 130-US Hospitals, #296, CC BY 4.0)
# ─────────────────────────────────────────────
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_ZIP_PATH = DATA_RAW / "diabetes_130.zip"
MAIN_CSV_NAME = "diabetic_data.csv"       # the one patient-level table
IDS_MAPPING_NAME = "IDS_mapping.csv"      # code -> text dictionary
MAIN_CSV_PATH = DATA_RAW / MAIN_CSV_NAME

# ─────────────────────────────────────────────
# Target
# ─────────────────────────────────────────────
RAW_TARGET_COL = "readmitted"     # values: '<30', '>30', 'NO'
TARGET_COL = "readmit_30d"        # engineered binary: 1 if '<30' else 0

# ─────────────────────────────────────────────
# Identifiers
# ─────────────────────────────────────────────
PATIENT_ID_COL = "patient_nbr"
ENCOUNTER_ID_COL = "encounter_id"

# ─────────────────────────────────────────────
# Cleaning constants
# ─────────────────────────────────────────────
MISSING_TOKEN = "?"
# discharge_disposition_id codes for expired / hospice — such patients
# cannot be readmitted, so including them leaks/poisons the target.
EXPIRED_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
# columns dropped outright (near-empty or leakage-prone identifiers)
DROP_COLUMNS = ["weight", "payer_code"]
# rare-category collapse threshold (fraction of rows) for high-cardinality cats
RARE_CATEGORY_MIN_FRAC = 0.01
OUTLIER_UPPER_QUANTILE = 0.99

# The 23 medication columns in this dataset (values: No/Steady/Up/Down)
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Numeric utilization/count columns present in the raw data
RAW_NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
# columns whose extreme upper tail we cap
OUTLIER_CAP_COLS = [
    "time_in_hospital", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient",
]

# ─────────────────────────────────────────────
# Risk tiers (quantile-based; see risk_scoring.py)
# ─────────────────────────────────────────────
TIER_LABELS = ["Low", "Medium", "High"]
TIER_QUANTILES = [0.0, 0.50, 0.80, 1.0]   # Low <P50, Medium P50-P80, High >=P80
