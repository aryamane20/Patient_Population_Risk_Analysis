"""
Healthcare FHIR Data Pipeline — Function Definitions
Mindbowser Learning Project — Day 14

This module holds ALL function definitions for the pipeline:
FHIR API calls, data flattening, feature engineering, de-identification,
data-quality validation, the pipeline orchestrator, and the rules-based
risk score.

It contains NO calling / execution code. Import these functions from
main.py (or a notebook) to run them.

Data source: HAPI FHIR Public Sandbox (hapi.fhir.org)
Note: In production this would connect to Epic/Cerner FHIR API
with SMART on FHIR authentication.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import warnings
import hashlib

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BASE_URL = "https://hapi.fhir.org/baseR4"
HEADERS = {"Accept": "application/fhir+json"}

# LOINC codes for key lab tests
# These are the lab values most predictive of readmission risk
LOINC_CODES = {
    "hba1c": "4548-4",           # HbA1c — diabetes control
    "glucose": "2345-7",          # Blood glucose
    "creatinine": "2160-0",       # Kidney function
    "sodium": "2951-2",           # Electrolyte — abnormal = dehydration risk
    "hemoglobin": "718-7",        # Anemia indicator
    "systolic_bp": "8480-6",      # Blood pressure systolic
    "diastolic_bp": "8462-4",     # Blood pressure diastolic
    "heart_rate": "8867-4",       # Heart rate
    "bmi": "39156-5"              # BMI
}

# ICD-10 prefixes for high-risk conditions
# These are the Elixhauser comorbidities most associated with readmission
HIGH_RISK_CONDITION_PREFIXES = {
    "diabetes": ["E10", "E11", "E13"],
    "heart_failure": ["I50"],
    "copd": ["J44", "J43"],
    "ckd": ["N18"],
    "hypertension": ["I10", "I11", "I12", "I13"],
    "coronary_artery_disease": ["I25"],
    "atrial_fibrillation": ["I48"],
    "depression": ["F32", "F33"],
    "cancer": ["C"],
    "stroke": ["I63", "I64"]
}

# ─────────────────────────────────────────────
# SECTION 1 — FHIR API CALLS
# ─────────────────────────────────────────────

def get_patients(count=20):
    """
    Fetch a list of patients from the FHIR server who actually
    have clinical data attached.

    Why not just query /Patient directly?
    On the HAPI public sandbox, /Patient?_sort=-_lastUpdated returns
    the most recently POSTed test records — which are almost always
    bare Patient resources with NO linked Conditions/Observations/
    Encounters. That produced a dataset of all-zero counts.

    Instead we discover patients by walking backwards from the
    Condition resource: any patient referenced by a Condition is
    guaranteed to have at least some clinical data. We then fetch
    the full Patient resource for each.

    In production: this would use FHIR Bulk Data export or query
    patients from a specific cohort (e.g., all patients discharged
    in the last 30 days) — where every patient has data by definition.
    """
    print(f"Discovering patients with clinical data (target: {count})...")

    # Step 1: pull a batch of Conditions and collect the patients they
    # point at. We over-fetch (count * 4) because many conditions share
    # the same patient.
    response = requests.get(
        f"{BASE_URL}/Condition",
        params={"_count": count * 4},
        headers=HEADERS
    )

    if response.status_code != 200:
        print(f"Error discovering patients via Condition: {response.status_code}")
        return []

    bundle = response.json()

    # Preserve order, de-duplicate patient IDs
    patient_ids = []
    seen = set()
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        ref = resource.get("subject", {}).get("reference", "")
        if ref.startswith("Patient/"):
            pid = ref.split("/", 1)[1]
            if pid and pid not in seen:
                seen.add(pid)
                patient_ids.append(pid)
        if len(patient_ids) >= count:
            break

    # Step 2: fetch the full Patient resource for each discovered ID
    patients = []
    for pid in patient_ids:
        resp = requests.get(f"{BASE_URL}/Patient/{pid}", headers=HEADERS)
        if resp.status_code != 200:
            continue
        resource = resp.json()
        if resource.get("resourceType") == "Patient":
            patients.append(resource)

    print(f"Retrieved {len(patients)} patients with clinical data")
    return patients


def get_conditions(patient_id):
    """
    Fetch all conditions (diagnoses) for a patient.
    Returns active conditions with ICD-10 or SNOMED codes.

    Clinical note: We want ALL conditions ever documented,
    not just current active ones — because resolved conditions
    still indicate disease history relevant to readmission risk.
    """
    response = requests.get(
        f"{BASE_URL}/Condition",
        params={
            "patient": patient_id,
            "_count": 100
        },
        headers=HEADERS
    )

    if response.status_code != 200:
        return []

    bundle = response.json()
    conditions = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue

        # Extract code information
        code_obj = resource.get("code", {})
        codings = code_obj.get("coding", [])

        for coding in codings:
            conditions.append({
                "patient_id": patient_id,
                "condition_id": resource.get("id", ""),
                "code": coding.get("code", ""),
                "display": coding.get("display", ""),
                "system": coding.get("system", ""),
                "clinical_status": (
                    resource.get("clinicalStatus", {})
                    .get("coding", [{}])[0]
                    .get("code", "unknown")
                ),
                "onset_date": resource.get("onsetDateTime", "")
            })

    return conditions


def get_observations(patient_id, loinc_codes=None):
    """
    Fetch lab results and vital signs for a patient.

    Clinical note: We fetch the most recent observations.
    For the readmission model, the TREND of lab values
    matters as much as the absolute value — a creatinine
    rising from 0.9 to 1.4 over 3 months is more concerning
    than a stable creatinine of 1.4.
    """
    # Note: we deliberately do NOT filter by category (e.g.
    # "laboratory,vital-signs") here. Many FHIR servers (including the
    # HAPI sandbox) store Observations without a category element, so
    # that filter silently drops every result. Pulling ALL Observations
    # and selecting by LOINC code already captures both labs AND vital
    # signs (systolic BP, BMI, heart rate are in LOINC_CODES), which is
    # exactly what the "add vital signs" extension is after — so no
    # category filter is needed.
    params = {
        "patient": patient_id,
        "_count": 50,
        "_sort": "-date"
    }

    # If specific LOINC codes requested, filter by them
    if loinc_codes:
        params["code"] = ",".join(loinc_codes)

    response = requests.get(
        f"{BASE_URL}/Observation",
        params=params,
        headers=HEADERS
    )

    if response.status_code != 200:
        return []

    bundle = response.json()
    observations = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue

        # Extract LOINC code
        codings = resource.get("code", {}).get("coding", [])
        loinc_code = ""
        display = ""
        for coding in codings:
            if "loinc" in coding.get("system", "").lower():
                loinc_code = coding.get("code", "")
                display = coding.get("display", "")
                break

        # Extract value — handle different value types
        value = None
        unit = ""

        if "valueQuantity" in resource:
            value = resource["valueQuantity"].get("value")
            unit = resource["valueQuantity"].get("unit", "")
        elif "valueCodeableConcept" in resource:
            codings = resource["valueCodeableConcept"].get("coding", [{}])
            value = codings[0].get("display", "") if codings else ""
        elif "valueString" in resource:
            value = resource["valueString"]

        # Extract interpretation (H=High, L=Low, N=Normal)
        interp_list = resource.get("interpretation", [])
        interpretation = ""
        if interp_list:
            interp_codings = interp_list[0].get("coding", [{}])
            interpretation = interp_codings[0].get("code", "") if interp_codings else ""

        observations.append({
            "patient_id": patient_id,
            "observation_id": resource.get("id", ""),
            "loinc_code": loinc_code,
            "display": display,
            "value": value,
            "unit": unit,
            "interpretation": interpretation,
            "effective_date": resource.get("effectiveDateTime", "")
        })

    return observations


def get_encounters(patient_id):
    """
    Fetch encounter history for a patient.

    Clinical note: We look at encounters in the last 2 years.
    Prior utilization is the strongest single predictor of
    future utilization. A patient with 5 hospitalizations
    in the last year is at dramatically higher risk than
    a patient with zero, regardless of current diagnosis.
    """
    response = requests.get(
        f"{BASE_URL}/Encounter",
        params={
            "patient": patient_id,
            "_count": 50,
            "_sort": "-date"
        },
        headers=HEADERS
    )

    if response.status_code != 200:
        return []

    bundle = response.json()
    encounters = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Encounter":
            continue

        # Extract encounter class (inpatient, outpatient, ED)
        enc_class = resource.get("class", {})
        class_code = enc_class.get("code", "unknown")

        # Extract period (admission and discharge dates)
        period = resource.get("period", {})
        start = period.get("start", "")
        end = period.get("end", "")

        # Calculate length of stay if both dates available
        length_of_stay = None
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                length_of_stay = (end_dt - start_dt).days
            except:
                pass

        encounters.append({
            "patient_id": patient_id,
            "encounter_id": resource.get("id", ""),
            "status": resource.get("status", ""),
            "class_code": class_code,
            "start_date": start,
            "end_date": end,
            "length_of_stay_days": length_of_stay
        })

    return encounters


def get_medication_count(patient_id):
    """
    Count active medications — polypharmacy is a readmission risk factor.
    Patients on 5+ medications have significantly higher readmission risk.
    """
    response = requests.get(
        f"{BASE_URL}/MedicationRequest",
        params={"patient": patient_id, "status": "active", "_count": 100},
        headers=HEADERS
    )

    if response.status_code != 200:
        return 0

    bundle = response.json()
    return len(bundle.get("entry", []))


# ─────────────────────────────────────────────
# SECTION 2 — DATA FLATTENING
# ─────────────────────────────────────────────

def flatten_patient(patient_resource):
    """
    Extract key fields from FHIR Patient resource.

    Note: In this function we extract PHI fields.
    These will be de-identified in Section 4 before
    any data leaves this pipeline.
    """
    patient_id = patient_resource.get("id", "")

    # Name
    names = patient_resource.get("name", [])
    family_name = ""
    given_name = ""
    if names:
        family_name = names[0].get("family", "")
        given_list = names[0].get("given", [])
        given_name = given_list[0] if given_list else ""

    # Date of birth
    dob = patient_resource.get("birthDate", "")

    # Calculate age from DOB
    age = None
    if dob:
        try:
            birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
            today = date.today()
            age = relativedelta(today, birth_date).years
        except:
            pass

    # Gender
    gender = patient_resource.get("gender", "unknown")

    # Address — extract state only (ZIP is PHI)
    addresses = patient_resource.get("address", [])
    state = ""
    zip_code = ""
    if addresses:
        state = addresses[0].get("state", "")
        zip_code = addresses[0].get("postalCode", "")

    # Phone
    telecoms = patient_resource.get("telecom", [])
    phone = ""
    for telecom in telecoms:
        if telecom.get("system") == "phone":
            phone = telecom.get("value", "")
            break

    return {
        "patient_id": patient_id,
        "family_name": family_name,        # PHI — will be removed
        "given_name": given_name,          # PHI — will be removed
        "date_of_birth": dob,              # PHI — will be replaced with age
        "age": age,
        "gender": gender,
        "state": state,                    # OK — state level is not PHI
        "zip_code": zip_code,             # PHI — will be removed
        "phone": phone                     # PHI — will be removed
    }


# ─────────────────────────────────────────────
# SECTION 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def check_high_risk_conditions(conditions_list):
    """
    Check which high-risk conditions a patient has.

    Clinical note: These conditions — the Elixhauser comorbidities —
    are validated in the clinical literature as the strongest
    diagnosis-based predictors of hospital readmission, mortality,
    and healthcare resource utilization.
    """
    features = {}

    all_codes = [c.get("code", "") for c in conditions_list]

    for condition_name, prefixes in HIGH_RISK_CONDITION_PREFIXES.items():
        has_condition = any(
            any(code.startswith(prefix) for prefix in prefixes)
            for code in all_codes
        )
        features[f"has_{condition_name}"] = int(has_condition)

    return features


def get_latest_lab_value(observations_list, loinc_code):
    """
    Get the most recent value for a specific lab test.

    Clinical note: We want the value closest to the index
    encounter — the one that reflects the patient's state
    at the time of discharge. A value from 6 months ago
    is less predictive than one from last week.
    """
    relevant = [
        obs for obs in observations_list
        if obs.get("loinc_code") == loinc_code
        and obs.get("value") is not None
        and obs.get("effective_date", "")
    ]

    if not relevant:
        return None, None, None

    # Sort by date descending — most recent first
    try:
        relevant.sort(
            key=lambda x: x.get("effective_date", ""),
            reverse=True
        )
    except:
        pass

    latest = relevant[0]
    value = latest.get("value")
    interpretation = latest.get("interpretation", "")
    date_str = latest.get("effective_date", "")

    # Convert to float if numeric
    try:
        value = float(value)
    except (TypeError, ValueError):
        pass

    return value, interpretation, date_str


def engineer_features(patient_flat, conditions, observations, encounters):
    """
    Build the complete feature vector for this patient.

    This is where clinical domain knowledge meets data science.
    Every feature here has a clinical rationale for why it
    might predict readmission risk.
    """
    features = {}
    patient_id = patient_flat["patient_id"]

    # ── Demographics ──
    features["age"] = patient_flat.get("age")
    features["gender_female"] = int(patient_flat.get("gender", "") == "female")
    features["gender_male"] = int(patient_flat.get("gender", "") == "male")

    # Age risk flag — patients over 65 have higher readmission risk
    # (Medicare patient, multiple chronic conditions more likely)
    age = patient_flat.get("age")
    features["age_over_65"] = int(age >= 65) if age is not None else None
    features["age_over_80"] = int(age >= 80) if age is not None else None

    # ── Condition burden ──
    active_conditions = [
        c for c in conditions
        if c.get("clinical_status") in ["active", "Active"]
    ]
    features["num_active_conditions"] = len(active_conditions)
    features["num_total_conditions"] = len(conditions)

    # High comorbidity flag — Elixhauser score proxy
    # More than 3 chronic conditions = significantly elevated risk
    features["high_comorbidity_burden"] = int(len(active_conditions) >= 3)

    # Individual high-risk condition flags
    condition_features = check_high_risk_conditions(conditions)
    features.update(condition_features)

    # ── Lab values ──
    # HbA1c — diabetes control
    # Clinical threshold: > 8.0% = poorly controlled
    hba1c_val, hba1c_interp, _ = get_latest_lab_value(observations, LOINC_CODES["hba1c"])
    features["hba1c_value"] = hba1c_val
    features["hba1c_abnormal"] = int(hba1c_interp == "H") if hba1c_interp else None
    features["hba1c_poor_control"] = (
        int(float(hba1c_val) > 8.0)
        if hba1c_val is not None and isinstance(hba1c_val, (int, float))
        else None
    )

    # Creatinine — kidney function
    # Clinical threshold: > 1.2 (female) or > 1.3 (male) = elevated
    creat_val, creat_interp, _ = get_latest_lab_value(observations, LOINC_CODES["creatinine"])
    features["creatinine_value"] = creat_val
    features["creatinine_abnormal"] = int(creat_interp == "H") if creat_interp else None

    # Sodium — electrolyte balance
    # Hyponatremia (low sodium) is strongly associated with readmission
    sodium_val, sodium_interp, _ = get_latest_lab_value(observations, LOINC_CODES["sodium"])
    features["sodium_value"] = sodium_val
    features["sodium_low"] = int(sodium_interp == "L") if sodium_interp else None

    # Hemoglobin — anemia
    hgb_val, hgb_interp, _ = get_latest_lab_value(observations, LOINC_CODES["hemoglobin"])
    features["hemoglobin_value"] = hgb_val
    features["anemia_flag"] = int(hgb_interp == "L") if hgb_interp else None

    # ── Vital signs ──
    systolic_val, systolic_interp, _ = get_latest_lab_value(observations, LOINC_CODES["systolic_bp"])
    features["systolic_bp"] = systolic_val
    features["hypertensive_urgency"] = (
        int(float(systolic_val) >= 180)
        if systolic_val is not None and isinstance(systolic_val, (int, float))
        else None
    )

    # BMI
    bmi_val, _, _ = get_latest_lab_value(observations, LOINC_CODES["bmi"])
    features["bmi"] = bmi_val
    features["obese"] = (
        int(float(bmi_val) >= 30)
        if bmi_val is not None and isinstance(bmi_val, (int, float))
        else None
    )

    # ── Medications ──
    # Polypharmacy (5+ active medications) is associated with higher
    # readmission, adverse drug events, and medication non-adherence.
    num_medications = get_medication_count(patient_id)
    features["num_active_medications"] = num_medications
    features["polypharmacy"] = int(num_medications >= 5)

    # ── Encounter history ──
    # Prior utilization is the single strongest predictor of future utilization

    total_encounters = len(encounters)
    features["total_encounters_ever"] = total_encounters

    # Count by encounter type
    inpatient_encounters = [
        e for e in encounters
        if e.get("class_code", "").upper() in ["IMP", "INPATIENT", "I"]
    ]
    ed_encounters = [
        e for e in encounters
        if e.get("class_code", "").upper() in ["EMER", "EMERGENCY", "ED", "E"]
    ]

    features["num_inpatient_encounters"] = len(inpatient_encounters)
    features["num_ed_encounters"] = len(ed_encounters)

    # High utilizer flag — 3+ inpatient admissions = very high risk
    features["high_utilizer"] = int(len(inpatient_encounters) >= 3)

    # Average length of stay for inpatient encounters
    los_values = [
        e.get("length_of_stay_days")
        for e in inpatient_encounters
        if e.get("length_of_stay_days") is not None
    ]
    features["avg_length_of_stay"] = (
        round(np.mean(los_values), 1) if los_values else None
    )

    # Long stay flag — LOS > 7 days indicates severity
    features["long_stay_history"] = int(
        any(los > 7 for los in los_values)
    ) if los_values else None

    # ── Composite risk indicators ──
    # These combine multiple signals into higher-order features

    # Diabetes + poor control = very high readmission risk
    has_diabetes = features.get("has_diabetes", 0)
    poor_hba1c = features.get("hba1c_poor_control")
    features["uncontrolled_diabetes"] = int(
        has_diabetes == 1 and poor_hba1c == 1
    ) if poor_hba1c is not None else None

    # Heart failure + anemia = very high readmission risk
    has_hf = features.get("has_heart_failure", 0)
    has_anemia = features.get("anemia_flag")
    features["heart_failure_with_anemia"] = int(
        has_hf == 1 and has_anemia == 1
    ) if has_anemia is not None else None

    # Number of abnormal lab values — overall disease severity proxy
    abnormal_flags = [
        features.get("hba1c_abnormal"),
        features.get("creatinine_abnormal"),
        features.get("sodium_low"),
        features.get("anemia_flag")
    ]
    features["num_abnormal_labs"] = sum(
        1 for f in abnormal_flags if f == 1
    )

    return features


# ─────────────────────────────────────────────
# SECTION 4 — DE-IDENTIFICATION
# ─────────────────────────────────────────────

def deidentify_patient(patient_flat, features):
    """
    Apply Safe Harbor de-identification.

    Remove or transform all 18 PHI identifiers.
    What remains is legally de-identified under HIPAA Safe Harbor
    and can be used without BAA restrictions.

    PHI fields being removed:
    1. Names → removed
    2. Geographic data below state → ZIP removed, state kept
    3. Dates (DOB) → replaced with age (calculated before removal)
    4. Phone numbers → removed
    5-18. All other identifiers → not present in this dataset

    What we keep:
    - Patient token (pseudonymized ID for longitudinal linking)
    - Age (not a PHI identifier)
    - Gender (not a PHI identifier)
    - State (state level is not PHI)
    - All clinical features (not PHI)
    """

    # Create a pseudonymized patient token
    # SHA-256 hash of the real patient ID
    # This allows longitudinal linking without revealing the real ID
    # Note: This is pseudonymization, not de-identification of the ID
    # The token table mapping real ID to token must be kept secure
    patient_token = hashlib.sha256(
        patient_flat["patient_id"].encode()
    ).hexdigest()[:16]

    deidentified = {
        # Pseudonymized ID — not PHI, enables longitudinal analysis
        "patient_token": patient_token,

        # Age — calculated from DOB before DOB was removed
        # Age under 90 is not PHI
        "age": features.get("age"),
        "age_over_65": features.get("age_over_65"),
        "age_over_80": features.get("age_over_80"),

        # Gender — not PHI
        "gender_female": features.get("gender_female"),
        "gender_male": features.get("gender_male"),

        # State — state level is not PHI
        "state": patient_flat.get("state"),

        # All clinical features — not PHI
        # Condition burden
        "num_active_conditions": features.get("num_active_conditions"),
        "num_total_conditions": features.get("num_total_conditions"),
        "high_comorbidity_burden": features.get("high_comorbidity_burden"),

        # Elixhauser conditions
        "has_diabetes": features.get("has_diabetes"),
        "has_heart_failure": features.get("has_heart_failure"),
        "has_copd": features.get("has_copd"),
        "has_ckd": features.get("has_ckd"),
        "has_hypertension": features.get("has_hypertension"),
        "has_coronary_artery_disease": features.get("has_coronary_artery_disease"),
        "has_atrial_fibrillation": features.get("has_atrial_fibrillation"),
        "has_depression": features.get("has_depression"),
        "has_cancer": features.get("has_cancer"),
        "has_stroke": features.get("has_stroke"),

        # Lab values
        "hba1c_value": features.get("hba1c_value"),
        "hba1c_abnormal": features.get("hba1c_abnormal"),
        "hba1c_poor_control": features.get("hba1c_poor_control"),
        "creatinine_value": features.get("creatinine_value"),
        "creatinine_abnormal": features.get("creatinine_abnormal"),
        "sodium_value": features.get("sodium_value"),
        "sodium_low": features.get("sodium_low"),
        "hemoglobin_value": features.get("hemoglobin_value"),
        "anemia_flag": features.get("anemia_flag"),

        # Vital signs
        "systolic_bp": features.get("systolic_bp"),
        "hypertensive_urgency": features.get("hypertensive_urgency"),
        "bmi": features.get("bmi"),
        "obese": features.get("obese"),

        # Medications
        "num_active_medications": features.get("num_active_medications"),
        "polypharmacy": features.get("polypharmacy"),

        # Encounter history
        "total_encounters_ever": features.get("total_encounters_ever"),
        "num_inpatient_encounters": features.get("num_inpatient_encounters"),
        "num_ed_encounters": features.get("num_ed_encounters"),
        "high_utilizer": features.get("high_utilizer"),
        "avg_length_of_stay": features.get("avg_length_of_stay"),
        "long_stay_history": features.get("long_stay_history"),

        # Composite features
        "uncontrolled_diabetes": features.get("uncontrolled_diabetes"),
        "heart_failure_with_anemia": features.get("heart_failure_with_anemia"),
        "num_abnormal_labs": features.get("num_abnormal_labs"),
    }

    # PHI fields explicitly NOT included:
    # - patient_id (real MRN) → replaced with patient_token
    # - family_name → removed
    # - given_name → removed
    # - date_of_birth → replaced with age
    # - zip_code → removed
    # - phone → removed

    return deidentified


# ─────────────────────────────────────────────
# SECTION 5 — DATA QUALITY VALIDATION
# ─────────────────────────────────────────────

def validate_data_quality(df):
    """
    Run data quality checks on the final dataset.

    In production this would be a more comprehensive DQA
    but this gives you the pattern for the key checks.
    """
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    total_records = len(df)
    print(f"\nTotal patients processed: {total_records}")

    # Completeness check for critical features
    critical_features = [
        "age",
        "num_active_conditions",
        "num_inpatient_encounters",
        "num_ed_encounters"
    ]

    print("\n--- COMPLETENESS (Critical Features) ---")
    for feature in critical_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            pct = round(non_null * 100 / total_records, 1)
            status = "✅" if pct >= 90 else "⚠️" if pct >= 70 else "❌"
            print(f"{status} {feature}: {pct}% complete ({non_null}/{total_records})")

    # Lab value completeness
    print("\n--- COMPLETENESS (Lab Values) ---")
    lab_features = [
        "hba1c_value",
        "creatinine_value",
        "sodium_value",
        "hemoglobin_value",
        "systolic_bp"
    ]
    for feature in lab_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            pct = round(non_null * 100 / total_records, 1)
            status = "✅" if pct >= 70 else "⚠️" if pct >= 40 else "❌"
            print(f"{status} {feature}: {pct}% complete ({non_null}/{total_records})")

    # Validity checks
    print("\n--- VALIDITY CHECKS ---")

    # Age range check
    if "age" in df.columns:
        invalid_age = df[
            df["age"].notna() & ((df["age"] < 0) | (df["age"] > 120))
        ]
        print(f"{'✅' if len(invalid_age) == 0 else '❌'} Invalid ages (outside 0-120): {len(invalid_age)}")

    # HbA1c range check — biologically possible range is 3-20%
    if "hba1c_value" in df.columns:
        invalid_hba1c = df[
            df["hba1c_value"].notna() &
            ((df["hba1c_value"] < 3) | (df["hba1c_value"] > 20))
        ]
        print(f"{'✅' if len(invalid_hba1c) == 0 else '❌'} Invalid HbA1c values (outside 3-20%): {len(invalid_hba1c)}")

    # Negative encounter counts check
    if "num_inpatient_encounters" in df.columns:
        negative_enc = df[
            df["num_inpatient_encounters"].notna() &
            (df["num_inpatient_encounters"] < 0)
        ]
        print(f"{'✅' if len(negative_enc) == 0 else '❌'} Negative encounter counts: {len(negative_enc)}")

    # Distribution summary
    print("\n--- DISTRIBUTION SUMMARY ---")

    if "age" in df.columns and df["age"].notna().any():
        print(f"Age: mean={df['age'].mean():.1f}, "
              f"median={df['age'].median():.0f}, "
              f"range={df['age'].min():.0f}-{df['age'].max():.0f}")

    if "num_active_conditions" in df.columns:
        print(f"Active conditions: mean={df['num_active_conditions'].mean():.1f}, "
              f"max={df['num_active_conditions'].max():.0f}")

    # Condition prevalence
    print("\n--- CONDITION PREVALENCE ---")
    condition_cols = [c for c in df.columns if c.startswith("has_")]
    for col in condition_cols:
        if col in df.columns:
            prevalence = df[col].mean() * 100 if df[col].notna().any() else 0
            print(f"  {col.replace('has_', '')}: {prevalence:.1f}%")

    # High risk flag summary
    print("\n--- RISK INDICATORS ---")
    if "high_utilizer" in df.columns:
        high_util = df["high_utilizer"].sum()
        print(f"High utilizers (3+ inpatient admits): "
              f"{high_util} ({high_util*100/total_records:.1f}%)")

    if "high_comorbidity_burden" in df.columns:
        high_comorbid = df["high_comorbidity_burden"].sum()
        print(f"High comorbidity burden (3+ conditions): "
              f"{high_comorbid} ({high_comorbid*100/total_records:.1f}%)")

    if "uncontrolled_diabetes" in df.columns:
        uncontrolled = df["uncontrolled_diabetes"].sum()
        print(f"Uncontrolled diabetes: "
              f"{uncontrolled} ({uncontrolled*100/total_records:.1f}%)")

    print("\n" + "="*60)


# ─────────────────────────────────────────────
# SECTION 6 — PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(num_patients=15):
    """
    Run the extraction → feature-engineering → de-identification →
    validation stages and return the de-identified DataFrame.

    Note: this function does NOT export a CSV or compute the risk
    score. Those steps live in main.py so that risk stratification
    can be applied before the file is written (keeping calling code
    separate from function definitions).

    In production this would be triggered by:
    - An ADT A03 discharge event (real-time mode)
    - A scheduled batch job (nightly mode)

    For this learning exercise we pull a fixed number
    of patients from the public FHIR sandbox.
    """

    print("="*60)
    print("HEALTHCARE FHIR DATA PIPELINE")
    print("Mindbowser Learning Project — Day 14")
    print("="*60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 1: Fetch patients ──
    patients = get_patients(count=num_patients)

    if not patients:
        print("No patients retrieved. Check FHIR server connection.")
        return None

    # ── Step 2: Process each patient ──
    all_records = []
    failed_patients = []

    print(f"\nProcessing {len(patients)} patients...")
    print("-" * 40)

    for i, patient_resource in enumerate(patients):
        patient_id = patient_resource.get("id", "unknown")

        try:
            # Flatten patient demographics
            patient_flat = flatten_patient(patient_resource)

            # Fetch clinical data
            conditions = get_conditions(patient_id)
            observations = get_observations(patient_id)
            encounters = get_encounters(patient_id)

            # Engineer features
            features = engineer_features(
                patient_flat,
                conditions,
                observations,
                encounters
            )

            # De-identify
            deidentified = deidentify_patient(patient_flat, features)

            all_records.append(deidentified)

            # Progress indicator
            print(f"[{i+1}/{len(patients)}] Patient {patient_id[:8]}... "
                  f"✅ {len(conditions)} conditions, "
                  f"{len(observations)} observations, "
                  f"{len(encounters)} encounters")

        except Exception as e:
            failed_patients.append(patient_id)
            print(f"[{i+1}/{len(patients)}] Patient {patient_id[:8]}... "
                  f"❌ Error: {str(e)[:50]}")

    if not all_records:
        print("\nNo records successfully processed.")
        return None

    # ── Step 3: Create DataFrame ──
    df = pd.DataFrame(all_records)

    print(f"\n{'='*40}")
    print(f"Pipeline complete: {len(all_records)} patients processed successfully")
    if failed_patients:
        print(f"Failed: {len(failed_patients)} patients")

    # ── Step 4: Data quality validation ──
    validate_data_quality(df)

    return df


# ─────────────────────────────────────────────
# SECTION 7 — SIMPLE RISK STRATIFICATION
# ─────────────────────────────────────────────

def calculate_simple_risk_score(row):
    """
    Simple additive risk score based on known risk factors.
    This is NOT a trained ML model — it's a rules-based score
    to demonstrate the concept before we train a proper model.

    In production: replace this with your trained model's
    predict_proba() output.
    """
    score = 0

    # Prior utilization — strongest predictor
    score += min((row.get("num_inpatient_encounters") or 0) * 15, 30)
    score += min((row.get("num_ed_encounters") or 0) * 5, 15)

    # High comorbidity burden
    score += (row.get("high_comorbidity_burden") or 0) * 10

    # Specific high-risk conditions
    score += (row.get("has_heart_failure") or 0) * 15
    score += (row.get("has_copd") or 0) * 10
    score += (row.get("has_ckd") or 0) * 10
    score += (row.get("uncontrolled_diabetes") or 0) * 12

    # Polypharmacy — 5+ active medications
    score += (row.get("polypharmacy") or 0) * 8

    # Abnormal lab values
    score += (row.get("num_abnormal_labs") or 0) * 8

    # Age risk
    score += (row.get("age_over_80") or 0) * 10
    score += (row.get("age_over_65") or 0) * 5

    # Cap at 100
    return min(score, 100)
