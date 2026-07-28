"""
ingest.py — the data-source abstraction layer.

WHY THIS EXISTS (production-readiness):
The model and everything downstream depend only on the `DataSource.load()`
contract — a DataFrame with the raw Diabetes-130 columns. Today that's a CSV;
in production it's a live FHIR/Epic feed. Swapping sources means writing one
new class, not touching the model. That is the whole point of this file.

Sources:
- CSVSource   : downloads + reads the real UCI Diabetes-130 dataset (used now).
- FHIRSource  : documented stub that reuses the Day-14 FHIR pipeline (prod).
"""

from __future__ import annotations

import io
import zipfile
from typing import Protocol

import pandas as pd
import requests

from . import config


class DataSource(Protocol):
    """Anything that can produce a raw patient-encounter DataFrame."""

    def load(self) -> pd.DataFrame:
        ...


class CSVSource:
    """
    Real data source: UCI Diabetes 130-US Hospitals.

    Downloads the zip once into data/raw/ (if absent), extracts the two CSVs,
    and returns the main patient-level table `diabetic_data.csv`.
    """

    def __init__(self, url: str = config.UCI_ZIP_URL,
                 csv_path=config.MAIN_CSV_PATH,
                 zip_path=config.RAW_ZIP_PATH):
        self.url = url
        self.csv_path = csv_path
        self.zip_path = zip_path

    def _download_and_extract(self) -> None:
        print(f"Downloading Diabetes-130 dataset from UCI...\n  {self.url}")
        resp = requests.get(self.url, timeout=120)
        resp.raise_for_status()
        self.zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            print(f"  Zip contents: {names}")
            # The archive nests the CSVs; extract whatever matches our targets.
            for member in names:
                base = member.split("/")[-1]
                if base in (config.MAIN_CSV_NAME, config.IDS_MAPPING_NAME):
                    with zf.open(member) as src:
                        (config.DATA_RAW / base).write_bytes(src.read())
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Expected {config.MAIN_CSV_NAME} in the archive but did not "
                f"find it. Archive members: {names}"
            )
        print(f"  Extracted -> {self.csv_path}")

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            self._download_and_extract()
        else:
            print(f"Using cached dataset at {self.csv_path}")
        # Read '?' as-is here (string); clean.py converts it to NaN so the
        # missing-value policy lives in one place.
        df = pd.read_csv(self.csv_path, dtype=str, na_filter=False)
        # Restore numeric dtypes for the count columns
        numeric_like = config.RAW_NUMERIC_COLS + [
            config.ENCOUNTER_ID_COL, config.PATIENT_ID_COL,
            "admission_type_id", "discharge_disposition_id",
            "admission_source_id",
        ]
        for col in numeric_like:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
        return df


class FHIRSource:
    """
    PRODUCTION stub — reuse the Day-14 FHIR pipeline (root pipeline_functions.py).

    In production this would:
      1. Pull Patient / Condition / Encounter / Observation from a FHIR endpoint
         (get_patients / get_conditions / get_encounters).
      2. Map those resources into the SAME raw column schema CSVSource returns,
         so clean.py + features.py work unchanged (the "feature contract").
      3. De-identify via deidentify_patient() before persisting to any ML store.

    Real-time admits (HL7 ADT A01/A03) would push single encounters through the
    same path. Left unimplemented on purpose for the Day-25 backend.
    """

    def load(self) -> pd.DataFrame:
        raise NotImplementedError(
            "FHIRSource is the production adapter. Wire it to "
            "pipeline_functions.py (get_patients/get_conditions/get_encounters) "
            "and map resources to the Diabetes-130 column schema. For Day 25 "
            "use CSVSource."
        )


def load_raw(source: DataSource | None = None) -> pd.DataFrame:
    """Convenience entry point. Defaults to the real CSV source."""
    source = source or CSVSource()
    return source.load()


if __name__ == "__main__":
    df = load_raw()
    print("\n--- Ingest smoke check ---")
    print("Shape:", df.shape)
    print("Target value counts:")
    print(df[config.RAW_TARGET_COL].value_counts())
