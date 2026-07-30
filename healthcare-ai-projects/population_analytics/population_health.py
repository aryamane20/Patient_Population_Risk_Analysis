"""
Population Health Analytics Platform — Engine (Function Definitions)
Day 19 — Mindbowser Healthcare AI Learning

This module holds ALL function definitions for the population health
analytics engine:
1. Patient population simulation (realistic distributions)
2. Risk stratification (predictive + acuity-based)
3. HEDIS measure calculation
4. Care gap identification
5. Panel management prioritization
6. Population-level analytics and reporting
7. Disease-specific cohort analysis
8. Outreach effectiveness simulation

It contains NO calling / execution code. Import these functions from
run_population_health.py (the entry point) or population_dashboard.py
(the Streamlit app).
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SECTION 1 — POPULATION DATA GENERATION
# ─────────────────────────────────────────────

def generate_population(n_patients: int = 10000,
                        random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic patient population for a
    mid-size regional health system.

    Distributions based on national averages:
    - Diabetes prevalence: ~11% of US adults
    - Hypertension prevalence: ~45% of US adults
    - Medicare age distribution
    - Urban/rural mix
    """

    np.random.seed(random_seed)
    n = n_patients

    print(f"Generating population of {n:,} patients...")

    # ── Demographics ──
    age = np.random.normal(58, 16, n).clip(18, 95).round(0).astype(int)
    gender = np.random.choice(['M', 'F'], n, p=[0.48, 0.52])
    race = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
        n, p=[0.62, 0.16, 0.12, 0.06, 0.04]
    )

    # Insurance/payer type
    insurance = np.where(
        age >= 65,
        np.random.choice(
            ['Medicare', 'Medicare_Advantage'],
            n, p=[0.55, 0.45]
        ),
        np.random.choice(
            ['Commercial', 'Medicaid', 'Self_pay', 'Medicare'],
            n, p=[0.55, 0.25, 0.10, 0.10]
        )
    )

    # ── Chronic Conditions ──
    # Diabetes (higher in older patients)
    diabetes_prob = np.where(
        age >= 65, 0.18,
        np.where(age >= 45, 0.12, 0.04)
    )
    has_diabetes = np.random.binomial(1, diabetes_prob)

    # Hypertension
    htn_prob = np.where(
        age >= 65, 0.70,
        np.where(age >= 45, 0.45, 0.15)
    )
    has_hypertension = np.random.binomial(1, htn_prob)

    # Heart disease
    heart_prob = np.where(
        age >= 65, 0.28,
        np.where(age >= 45, 0.12, 0.03)
    )
    has_heart_disease = np.random.binomial(1, heart_prob)

    # COPD
    copd_prob = np.where(age >= 55, 0.15, 0.03)
    has_copd = np.random.binomial(1, copd_prob)

    # CKD
    ckd_prob = np.where(
        has_diabetes == 1, 0.30,
        np.where(has_hypertension == 1, 0.20, 0.05)
    )
    has_ckd = np.random.binomial(1, ckd_prob)

    # Cancer history
    cancer_prob = np.where(age >= 50, 0.08, 0.02)
    has_cancer_history = np.random.binomial(1, cancer_prob)

    # Mental health
    has_depression = np.random.binomial(1, 0.12, n)

    # Number of chronic conditions (Elixhauser proxy)
    num_conditions = (has_diabetes + has_hypertension +
                      has_heart_disease + has_copd +
                      has_ckd + has_cancer_history +
                      has_depression)

    # ── Lab Values ──
    # HbA1c — only meaningful for diabetics
    hba1c = np.where(
        has_diabetes,
        np.random.normal(8.1, 1.6, n).clip(5.5, 15.0),
        np.random.normal(5.4, 0.3, n).clip(4.5, 6.4)
    ).round(1)

    # Blood pressure
    systolic_bp = np.where(
        has_hypertension,
        np.random.normal(142, 18, n).clip(90, 200),
        np.random.normal(118, 12, n).clip(90, 160)
    ).round(0).astype(int)

    diastolic_bp = (systolic_bp * 0.62 +
                    np.random.normal(0, 5, n)).round(0).astype(int).clip(50, 120)

    # LDL cholesterol
    ldl = np.where(
        has_heart_disease,
        np.random.normal(105, 28, n).clip(40, 220),
        np.random.normal(118, 30, n).clip(40, 250)
    ).round(0).astype(int)

    # BMI
    bmi = np.random.normal(29.5, 6.5, n).clip(16, 55).round(1)

    # ── Utilization History ──
    ed_visits_12m = np.random.poisson(
        0.3 + 0.8 * (num_conditions / 3).clip(0, 1), n
    ).clip(0, 10)

    inpatient_admits_12m = np.random.poisson(
        0.1 + 0.5 * (num_conditions / 3).clip(0, 1), n
    ).clip(0, 8)

    # Primary care visits
    pcp_visits_12m = np.random.poisson(
        1.5 + num_conditions * 0.8, n
    ).clip(0, 20)

    # ── Care Received (for HEDIS gaps) ──
    # Days since last HbA1c test (for diabetics)
    days_since_hba1c = np.where(
        has_diabetes,
        np.random.choice(
            [30, 90, 180, 270, 365, 548, 730, 9999],
            n,
            p=[0.05, 0.15, 0.20, 0.15, 0.20, 0.10, 0.08, 0.07]
        ),
        9999  # Non-diabetics don't need HbA1c
    )

    # Days since last BP reading
    days_since_bp = np.random.choice(
        [7, 30, 90, 180, 365, 548, 9999],
        n,
        p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.07, 0.03]
    )

    # Colorectal cancer screening (age 45-75)
    colonoscopy_due = np.where(
        (age >= 45) & (age <= 75),
        np.random.binomial(1, 0.35, n),  # 35% have gap
        0
    )

    # Mammogram (women 50-74)
    mammogram_due = np.where(
        (gender == 'F') & (age >= 50) & (age <= 74),
        np.random.binomial(1, 0.30, n),  # 30% have gap
        0
    )

    # Statin therapy for eligible patients
    statin_eligible = has_heart_disease | (has_diabetes & (age >= 40))
    statin_prescribed = np.where(
        statin_eligible,
        np.random.binomial(1, 0.72, n),  # 72% on statin if eligible
        0
    )
    statin_gap = np.where(
        statin_eligible & (statin_prescribed == 0),
        1, 0
    )

    # Medication adherence (PDC — proportion of days covered)
    # PDC >= 0.80 = adherent
    diabetes_med_pdc = np.where(
        has_diabetes,
        np.random.beta(5, 2, n),  # Most patients somewhat adherent
        1.0
    ).round(2)

    htn_med_pdc = np.where(
        has_hypertension,
        np.random.beta(4.5, 2, n),
        1.0
    ).round(2)

    statin_pdc = np.where(
        statin_prescribed == 1,
        np.random.beta(4, 2.5, n),
        1.0
    ).round(2)

    # Annual wellness visit
    awv_completed = np.where(
        age >= 65,
        np.random.binomial(1, 0.55, n),  # 55% completion rate for Medicare
        np.random.binomial(1, 0.35, n)
    )

    # ── Engagement Factors ──
    # MyChart portal enrollment (digital engagement)
    mychart_enrolled = np.random.binomial(1, 0.58, n)

    # Preferred contact method
    contact_preference = np.random.choice(
        ['Phone', 'Text', 'Email', 'Portal'],
        n, p=[0.40, 0.25, 0.20, 0.15]
    )

    # ── SDOH Risk ──
    transportation_barrier = np.random.binomial(1, 0.15, n)
    food_insecurity = np.random.binomial(1, 0.12, n)
    housing_instability = np.random.binomial(1, 0.08, n)
    social_isolation = np.random.binomial(1, 0.18, n)

    sdoh_risk_score = (transportation_barrier + food_insecurity +
                       housing_instability + social_isolation)

    # ── Last Interactions ──
    last_pcp_visit_days = np.random.choice(
        [30, 90, 180, 365, 548, 730, 9999],
        n,
        p=[0.15, 0.25, 0.25, 0.20, 0.08, 0.05, 0.02]
    )

    # ── Patient Identifiers ──
    patient_tokens = [f'PT{i:08d}' for i in range(n)]

    df = pd.DataFrame({
        'patient_token': patient_tokens,

        # Demographics
        'age': age,
        'gender': gender,
        'race': race,
        'insurance': insurance,

        # Conditions
        'has_diabetes': has_diabetes,
        'has_hypertension': has_hypertension,
        'has_heart_disease': has_heart_disease,
        'has_copd': has_copd,
        'has_ckd': has_ckd,
        'has_cancer_history': has_cancer_history,
        'has_depression': has_depression,
        'num_conditions': num_conditions,

        # Lab values
        'hba1c': hba1c,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'ldl': ldl,
        'bmi': bmi,

        # Utilization
        'ed_visits_12m': ed_visits_12m,
        'inpatient_admits_12m': inpatient_admits_12m,
        'pcp_visits_12m': pcp_visits_12m,

        # Care received / gaps
        'days_since_hba1c': days_since_hba1c,
        'days_since_bp': days_since_bp,
        'colonoscopy_due': colonoscopy_due,
        'mammogram_due': mammogram_due,
        'statin_eligible': statin_eligible.astype(int),
        'statin_prescribed': statin_prescribed,
        'statin_gap': statin_gap,

        # Medication adherence (PDC)
        'diabetes_med_pdc': diabetes_med_pdc,
        'htn_med_pdc': htn_med_pdc,
        'statin_pdc': statin_pdc,

        # Wellness
        'awv_completed': awv_completed,

        # Engagement
        'mychart_enrolled': mychart_enrolled,
        'contact_preference': contact_preference,

        # SDOH
        'transportation_barrier': transportation_barrier,
        'food_insecurity': food_insecurity,
        'housing_instability': housing_instability,
        'social_isolation': social_isolation,
        'sdoh_risk_score': sdoh_risk_score,

        # Recency
        'last_pcp_visit_days': last_pcp_visit_days
    })

    print(f"Population generated:")
    print(f"  Total patients:     {n:,}")
    print(f"  Diabetic:           {has_diabetes.sum():,} ({has_diabetes.mean()*100:.1f}%)")
    print(f"  Hypertensive:       {has_hypertension.sum():,} ({has_hypertension.mean()*100:.1f}%)")
    print(f"  Heart disease:      {has_heart_disease.sum():,} ({has_heart_disease.mean()*100:.1f}%)")
    print(f"  Age 65+:            {(age>=65).sum():,} ({(age>=65).mean()*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# SECTION 2 — RISK STRATIFICATION
# ─────────────────────────────────────────────

def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate population risk scores.

    Uses a composite scoring approach combining:
    1. Chronic condition burden (Elixhauser-based)
    2. Recent utilization (strongest predictor)
    3. Lab value control (clinical risk)
    4. SDOH risk (social vulnerability)
    5. Engagement (likelihood to respond to outreach)

    Outputs: predicted_cost_tier, risk_score, risk_tier
    """

    df = df.copy()

    # ── Component 1: Condition Burden Score (0-40) ──
    condition_score = (
        df['has_heart_disease'] * 12 +    # Highest cost condition
        df['has_diabetes'] * 10 +          # High chronic cost
        df['has_copd'] * 10 +              # High exacerbation risk
        df['has_ckd'] * 8 +               # Progression risk
        df['has_cancer_history'] * 6 +    # Treatment costs
        df['has_hypertension'] * 4 +      # Common, manageable
        df['has_depression'] * 4 +        # Affects compliance
        (df['num_conditions'] >= 3) * 6   # Polypharmacy/complexity bonus
    ).clip(0, 40)

    # ── Component 2: Utilization Score (0-30) ──
    utilization_score = (
        df['inpatient_admits_12m'] * 10 +   # Each admit = high risk
        df['ed_visits_12m'] * 5 +           # ED = high cost/risk
        (df['last_pcp_visit_days'] > 365) * 8  # Lost to follow-up
    ).clip(0, 30)

    # ── Component 3: Clinical Control Score (0-20) ──
    clinical_score = (
        # Uncontrolled diabetes
        ((df['has_diabetes']) & (df['hba1c'] > 9.0)) * 8 +
        ((df['has_diabetes']) & (df['hba1c'] > 8.0) &
         (df['hba1c'] <= 9.0)) * 4 +

        # Uncontrolled hypertension
        ((df['has_hypertension']) &
         (df['systolic_bp'] > 150)) * 6 +
        ((df['has_hypertension']) &
         (df['systolic_bp'] > 140) &
         (df['systolic_bp'] <= 150)) * 3 +

        # Statin gap in high-risk patient
        df['statin_gap'] * 4 +

        # Medication non-adherence
        ((df['has_diabetes']) &
         (df['diabetes_med_pdc'] < 0.80)) * 4 +
        ((df['has_hypertension']) &
         (df['htn_med_pdc'] < 0.80)) * 4
    ).clip(0, 20)

    # ── Component 4: SDOH Risk Score (0-10) ──
    sdoh_score = (df['sdoh_risk_score'] * 2.5).clip(0, 10)

    # ── Composite Risk Score (0-100) ──
    df['risk_score_raw'] = (
        condition_score +
        utilization_score +
        clinical_score +
        sdoh_score
    )

    # Normalize to 0-100 scale
    max_score = df['risk_score_raw'].max()
    df['risk_score'] = (
        (df['risk_score_raw'] / max_score * 100)
    ).round(1)

    # ── Risk Tiers ──
    df['risk_tier'] = pd.cut(
        df['risk_score'],
        bins=[0, 20, 40, 65, 100],
        labels=['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH'],
        include_lowest=True
    )

    # ── Predicted Cost Tier ──
    # Based on CMS actuarial categories
    df['predicted_cost_tier'] = pd.cut(
        df['risk_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Tier1_Low', 'Tier2_Med', 'Tier3_High', 'Tier4_Critical'],
        include_lowest=True
    )

    # ── Care Management Assignment ──
    df['care_management_level'] = np.select(
        [
            df['risk_tier'] == 'VERY_HIGH',
            df['risk_tier'] == 'HIGH',
            df['risk_tier'] == 'MODERATE',
            df['risk_tier'] == 'LOW'
        ],
        [
            'Intensive Case Management',
            'Care Management',
            'Disease Management Program',
            'Standard Preventive Care'
        ],
        default='Standard Preventive Care'
    )

    return df


# ─────────────────────────────────────────────
# SECTION 3 — HEDIS MEASURE CALCULATION
# ─────────────────────────────────────────────

def calculate_hedis_measures(df: pd.DataFrame) -> dict:
    """
    Calculate HEDIS quality measure performance.

    Returns performance rates for each applicable measure
    along with the denominator (eligible patients) and
    numerator (patients meeting the measure).

    These rates determine:
    - CMS Star Ratings for Medicare Advantage plans
    - Quality bonus payments
    - ACO shared savings calculations
    - Public quality report cards
    """

    measures = {}

    # ── Diabetes Measures ──

    # D1: HbA1c Testing (at least once in measurement year)
    diabetic_pts = df[df['has_diabetes'] == 1]
    hba1c_tested = diabetic_pts[diabetic_pts['days_since_hba1c'] <= 365]

    measures['CDC_HbA1c_Testing'] = {
        'name': 'Comprehensive Diabetes Care — HbA1c Testing',
        'denominator': len(diabetic_pts),
        'numerator': len(hba1c_tested),
        'rate': len(hba1c_tested) / len(diabetic_pts) * 100 if len(diabetic_pts) > 0 else 0,
        'national_average': 88.0,
        'target': 90.0,
        'gap_patients': diabetic_pts[
            diabetic_pts['days_since_hba1c'] > 365
        ]['patient_token'].tolist()
    }

    # D2: HbA1c Control (<8.0%)
    tested_pts = diabetic_pts[diabetic_pts['days_since_hba1c'] <= 365]
    controlled = tested_pts[tested_pts['hba1c'] < 8.0]

    measures['CDC_HbA1c_Control'] = {
        'name': 'Comprehensive Diabetes Care — HbA1c Control (<8.0%)',
        'denominator': len(tested_pts),
        'numerator': len(controlled),
        'rate': len(controlled) / len(tested_pts) * 100 if len(tested_pts) > 0 else 0,
        'national_average': 58.0,
        'target': 65.0,
        'gap_patients': tested_pts[
            tested_pts['hba1c'] >= 8.0
        ]['patient_token'].tolist()
    }

    # D3: Poor HbA1c Control (>9.0%) — lower is better
    poor_control = tested_pts[tested_pts['hba1c'] > 9.0]

    measures['CDC_Poor_HbA1c_Control'] = {
        'name': 'Comprehensive Diabetes Care — Poor HbA1c Control (>9.0%)',
        'denominator': len(tested_pts),
        'numerator': len(poor_control),
        'rate': len(poor_control) / len(tested_pts) * 100 if len(tested_pts) > 0 else 0,
        'national_average': 21.0,
        'target': 18.0,  # Lower is better
        'lower_is_better': True,
        'gap_patients': poor_control['patient_token'].tolist()
    }

    # ── Blood Pressure Measures ──

    # BP1: Controlling High Blood Pressure
    htn_pts = df[df['has_hypertension'] == 1]
    bp_controlled = htn_pts[
        (htn_pts['systolic_bp'] < 140) &
        (htn_pts['diastolic_bp'] < 90)
    ]

    measures['CBP_Blood_Pressure_Control'] = {
        'name': 'Controlling High Blood Pressure (<140/90)',
        'denominator': len(htn_pts),
        'numerator': len(bp_controlled),
        'rate': len(bp_controlled) / len(htn_pts) * 100 if len(htn_pts) > 0 else 0,
        'national_average': 62.0,
        'target': 70.0,
        'gap_patients': htn_pts[
            (htn_pts['systolic_bp'] >= 140) |
            (htn_pts['diastolic_bp'] >= 90)
        ]['patient_token'].tolist()
    }

    # ── Cancer Screening Measures ──

    # COL: Colorectal Cancer Screening
    col_eligible = df[(df['age'] >= 45) & (df['age'] <= 75)]
    col_screened = col_eligible[col_eligible['colonoscopy_due'] == 0]

    measures['COL_Colorectal_Screening'] = {
        'name': 'Colorectal Cancer Screening',
        'denominator': len(col_eligible),
        'numerator': len(col_screened),
        'rate': len(col_screened) / len(col_eligible) * 100 if len(col_eligible) > 0 else 0,
        'national_average': 68.0,
        'target': 75.0,
        'gap_patients': col_eligible[
            col_eligible['colonoscopy_due'] == 1
        ]['patient_token'].tolist()
    }

    # BCS: Breast Cancer Screening
    bcs_eligible = df[
        (df['gender'] == 'F') &
        (df['age'] >= 50) &
        (df['age'] <= 74)
    ]
    bcs_screened = bcs_eligible[bcs_eligible['mammogram_due'] == 0]

    measures['BCS_Breast_Cancer_Screening'] = {
        'name': 'Breast Cancer Screening (Mammogram)',
        'denominator': len(bcs_eligible),
        'numerator': len(bcs_screened),
        'rate': len(bcs_screened) / len(bcs_eligible) * 100 if len(bcs_eligible) > 0 else 0,
        'national_average': 72.0,
        'target': 78.0,
        'gap_patients': bcs_eligible[
            bcs_eligible['mammogram_due'] == 1
        ]['patient_token'].tolist()
    }

    # ── Medication Measures ──

    # SPC: Statin Use in Cardiovascular Disease
    statin_eligible_pts = df[df['statin_eligible'] == 1]
    statin_on_therapy = statin_eligible_pts[
        statin_eligible_pts['statin_prescribed'] == 1
    ]

    measures['SPC_Statin_Cardiovascular'] = {
        'name': 'Statin Use in Cardiovascular Disease',
        'denominator': len(statin_eligible_pts),
        'numerator': len(statin_on_therapy),
        'rate': len(statin_on_therapy) / len(statin_eligible_pts) * 100 if len(statin_eligible_pts) > 0 else 0,
        'national_average': 78.0,
        'target': 85.0,
        'gap_patients': statin_eligible_pts[
            statin_eligible_pts['statin_prescribed'] == 0
        ]['patient_token'].tolist()
    }

    # MAD: Medication Adherence for Diabetes
    dm_med_pts = df[df['has_diabetes'] == 1]
    dm_adherent = dm_med_pts[dm_med_pts['diabetes_med_pdc'] >= 0.80]

    measures['MAD_Diabetes_Adherence'] = {
        'name': 'Medication Adherence — Diabetes Medications (PDC≥80%)',
        'denominator': len(dm_med_pts),
        'numerator': len(dm_adherent),
        'rate': len(dm_adherent) / len(dm_med_pts) * 100 if len(dm_med_pts) > 0 else 0,
        'national_average': 83.0,
        'target': 87.0,
        'gap_patients': dm_med_pts[
            dm_med_pts['diabetes_med_pdc'] < 0.80
        ]['patient_token'].tolist()
    }

    # MAH: Medication Adherence for Hypertension
    htn_med_pts = df[df['has_hypertension'] == 1]
    htn_adherent = htn_med_pts[htn_med_pts['htn_med_pdc'] >= 0.80]

    measures['MAH_Hypertension_Adherence'] = {
        'name': 'Medication Adherence — Hypertension (PDC≥80%)',
        'denominator': len(htn_med_pts),
        'numerator': len(htn_adherent),
        'rate': len(htn_adherent) / len(htn_med_pts) * 100 if len(htn_med_pts) > 0 else 0,
        'national_average': 81.0,
        'target': 86.0,
        'gap_patients': htn_med_pts[
            htn_med_pts['htn_med_pdc'] < 0.80
        ]['patient_token'].tolist()
    }

    # AWV: Annual Wellness Visit (Medicare)
    medicare_pts = df[df['insurance'].isin(['Medicare', 'Medicare_Advantage'])]
    awv_pts = medicare_pts[medicare_pts['awv_completed'] == 1]

    measures['AWV_Annual_Wellness'] = {
        'name': 'Annual Wellness Visit Completion (Medicare)',
        'denominator': len(medicare_pts),
        'numerator': len(awv_pts),
        'rate': len(awv_pts) / len(medicare_pts) * 100 if len(medicare_pts) > 0 else 0,
        'national_average': 52.0,
        'target': 65.0,
        'gap_patients': medicare_pts[
            medicare_pts['awv_completed'] == 0
        ]['patient_token'].tolist()
    }

    return measures


# ─────────────────────────────────────────────
# SECTION 4 — CARE GAP IDENTIFICATION
# ─────────────────────────────────────────────

def identify_care_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify specific care gaps for each patient.

    A care gap = recommended service not received.

    Output: patient-level care gap list with:
    - Gap type
    - Priority (clinical urgency)
    - Recommended action
    - Outreach method
    """

    gaps = []

    for _, patient in df.iterrows():
        patient_gaps = []

        # ── Diabetes Gaps ──
        if patient['has_diabetes']:

            # HbA1c overdue
            if patient['days_since_hba1c'] > 365:
                patient_gaps.append({
                    'gap_type': 'HbA1c_Overdue',
                    'description': 'HbA1c test overdue (>12 months)',
                    'priority': 'HIGH',
                    'recommended_action': 'Order HbA1c lab — can be standing order or during next visit',
                    'gap_days': int(patient['days_since_hba1c'] - 365)
                })

            # Uncontrolled diabetes
            if patient['hba1c'] > 9.0 and patient['days_since_hba1c'] <= 365:
                patient_gaps.append({
                    'gap_type': 'Uncontrolled_Diabetes',
                    'description': f"Poor glycemic control — HbA1c {patient['hba1c']}%",
                    'priority': 'CRITICAL',
                    'recommended_action': 'Urgent care manager outreach — medication review and endocrinology referral',
                    'gap_days': 0
                })

            # Medication adherence gap
            if patient['diabetes_med_pdc'] < 0.80:
                patient_gaps.append({
                    'gap_type': 'DM_Med_Adherence',
                    'description': f"Diabetes medication non-adherence — PDC {patient['diabetes_med_pdc']:.0%}",
                    'priority': 'HIGH',
                    'recommended_action': 'Pharmacy outreach — barrier assessment, pill pack consideration',
                    'gap_days': 0
                })

        # ── Hypertension Gaps ──
        if patient['has_hypertension']:

            # Uncontrolled BP
            if (patient['systolic_bp'] >= 160 or
                patient['diastolic_bp'] >= 100):
                patient_gaps.append({
                    'gap_type': 'Uncontrolled_HTN_Severe',
                    'description': f"Severe uncontrolled BP: {patient['systolic_bp']}/{patient['diastolic_bp']}",
                    'priority': 'CRITICAL',
                    'recommended_action': 'Same-day nurse callback — check if patient symptomatic, urgent PCP appointment',
                    'gap_days': 0
                })

            elif (patient['systolic_bp'] >= 140 or
                  patient['diastolic_bp'] >= 90):
                patient_gaps.append({
                    'gap_type': 'Uncontrolled_HTN',
                    'description': f"Uncontrolled BP: {patient['systolic_bp']}/{patient['diastolic_bp']}",
                    'priority': 'HIGH',
                    'recommended_action': 'Schedule follow-up visit for medication adjustment',
                    'gap_days': 0
                })

            # Medication adherence gap
            if patient['htn_med_pdc'] < 0.80:
                patient_gaps.append({
                    'gap_type': 'HTN_Med_Adherence',
                    'description': f"Hypertension medication non-adherence — PDC {patient['htn_med_pdc']:.0%}",
                    'priority': 'HIGH',
                    'recommended_action': 'Pharmacy outreach — cost barrier assessment, generic alternatives',
                    'gap_days': 0
                })

        # ── Statin Gap ──
        if patient['statin_gap'] == 1:
            patient_gaps.append({
                'gap_type': 'Statin_Gap',
                'description': 'Statin not prescribed — eligible patient (CAD or diabetes age 40+)',
                'priority': 'HIGH',
                'recommended_action': 'PCP notification — add statin to medication regimen at next visit',
                'gap_days': 0
            })

        # ── Cancer Screening Gaps ──
        if patient['colonoscopy_due'] == 1:
            patient_gaps.append({
                'gap_type': 'Colorectal_Screening',
                'description': 'Colorectal cancer screening overdue',
                'priority': 'MODERATE',
                'recommended_action': 'Schedule colonoscopy or offer FIT test alternative',
                'gap_days': 0
            })

        if patient['mammogram_due'] == 1:
            patient_gaps.append({
                'gap_type': 'Mammogram_Screening',
                'description': 'Mammogram overdue',
                'priority': 'MODERATE',
                'recommended_action': 'Schedule mammogram — can offer mobile mammography if transportation barrier',
                'gap_days': 0
            })

        # ── Wellness Visit Gap ──
        if (patient['insurance'] in ['Medicare', 'Medicare_Advantage'] and
                patient['awv_completed'] == 0):
            patient_gaps.append({
                'gap_type': 'Annual_Wellness_Visit',
                'description': 'Annual Wellness Visit not completed this year',
                'priority': 'MODERATE',
                'recommended_action': 'Outreach to schedule AWV — covered at no cost to patient',
                'gap_days': 0
            })

        # ── Lost to Follow-up ──
        if patient['last_pcp_visit_days'] > 365:
            priority = 'HIGH' if patient['num_conditions'] >= 2 else 'MODERATE'
            patient_gaps.append({
                'gap_type': 'Lost_to_Followup',
                'description': f"No PCP visit in {patient['last_pcp_visit_days']} days",
                'priority': priority,
                'recommended_action': 'Reconnection outreach — identify barriers, offer telehealth',
                'gap_days': int(patient['last_pcp_visit_days'] - 365)
            })

        # ── SDOH Gaps ──
        if patient['sdoh_risk_score'] >= 2:
            patient_gaps.append({
                'gap_type': 'SDOH_High_Risk',
                'description': f"High SDOH risk score: {int(patient['sdoh_risk_score'])}/4",
                'priority': 'MODERATE',
                'recommended_action': 'Social work referral — address transportation, food, housing barriers',
                'gap_days': 0
            })

        # Store gaps for this patient
        for gap in patient_gaps:
            gaps.append({
                'patient_token': patient['patient_token'],
                'age': patient['age'],
                'risk_tier': patient['risk_tier'],
                'risk_score': patient['risk_score'],
                'num_conditions': patient['num_conditions'],
                'has_diabetes': patient['has_diabetes'],
                'has_hypertension': patient['has_hypertension'],
                'contact_preference': patient['contact_preference'],
                'mychart_enrolled': patient['mychart_enrolled'],
                **gap
            })

    if not gaps:
        return pd.DataFrame()

    gaps_df = pd.DataFrame(gaps)

    # Priority ordering for care coordinators
    priority_order = {
        'CRITICAL': 0,
        'HIGH': 1,
        'MODERATE': 2,
        'LOW': 3
    }
    gaps_df['priority_rank'] = gaps_df['priority'].map(priority_order)
    gaps_df = gaps_df.sort_values(
        ['priority_rank', 'risk_score'],
        ascending=[True, False]
    )

    return gaps_df


# ─────────────────────────────────────────────
# SECTION 5 — PANEL MANAGEMENT WORKLIST
# ─────────────────────────────────────────────

def generate_daily_worklist(df: pd.DataFrame,
                             gaps_df: pd.DataFrame,
                             care_coordinator_capacity: int = 30) -> pd.DataFrame:
    """
    Generate the care coordinator's daily action list.

    Prioritizes patients by:
    1. Clinical urgency (CRITICAL first)
    2. Risk score (highest risk within same priority)
    3. Number of open care gaps (more gaps = more to address)
    4. Days since last contact (long gap = higher priority)

    Limits to care coordinator's daily capacity.
    """

    if gaps_df.empty:
        return pd.DataFrame()

    # Aggregate gaps by patient
    patient_gap_summary = gaps_df.groupby('patient_token').agg(
        num_gaps=('gap_type', 'count'),
        highest_priority=('priority_rank', 'min'),
        gap_types=('gap_type', lambda x: ', '.join(sorted(x.unique()))),
        top_action=('recommended_action', 'first')
    ).reset_index()

    patient_gap_summary['highest_priority_label'] = patient_gap_summary['highest_priority'].map({
        0: 'CRITICAL', 1: 'HIGH', 2: 'MODERATE', 3: 'LOW'
    })

    # Merge with patient data
    worklist = df.merge(
        patient_gap_summary,
        on='patient_token',
        how='inner'
    )

    # Sort by priority then risk score
    worklist = worklist.sort_values(
        ['highest_priority', 'risk_score', 'num_gaps'],
        ascending=[True, False, False]
    )

    # Limit to daily capacity
    daily_list = worklist.head(care_coordinator_capacity).copy()

    # Add workflow fields
    daily_list['outreach_priority'] = range(1, len(daily_list) + 1)
    daily_list['suggested_channel'] = np.where(
        daily_list['highest_priority_label'] == 'CRITICAL',
        'PHONE — Same day',
        np.where(
            daily_list['mychart_enrolled'] == 1,
            'MyChart message',
            np.where(
                daily_list['contact_preference'] == 'Text',
                'Text message',
                'Phone call'
            )
        )
    )

    return daily_list[[
        'outreach_priority',
        'patient_token',
        'age',
        'risk_tier',
        'risk_score',
        'highest_priority_label',
        'num_gaps',
        'gap_types',
        'top_action',
        'suggested_channel',
        'num_conditions',
        'has_diabetes',
        'has_hypertension',
        'last_pcp_visit_days'
    ]]


# ─────────────────────────────────────────────
# SECTION 6 — POPULATION ANALYTICS
# ─────────────────────────────────────────────

def generate_population_report(df: pd.DataFrame,
                                measures: dict,
                                gaps_df: pd.DataFrame):
    """
    Generate the population health analytics report.

    This is what gets presented to the CMO and CDO monthly.
    Shows population health performance, trends,
    and opportunities for improvement.
    """

    print("\n" + "="*70)
    print("POPULATION HEALTH ANALYTICS REPORT")
    print(f"Health System: Regional Medical Center")
    print(f"Report Date: {datetime.now().strftime('%B %Y')}")
    print(f"Population: {len(df):,} attributed patients")
    print("="*70)

    # ── Population Overview ──
    print("\n📊 POPULATION OVERVIEW")
    print("-"*70)

    tier_dist = df['risk_tier'].value_counts()
    print(f"\nRisk Stratification:")
    for tier in ['VERY_HIGH', 'HIGH', 'MODERATE', 'LOW']:
        count = tier_dist.get(tier, 0)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {tier:<12} {count:>6,} ({pct:>5.1f}%) {bar}")

    # Cost concentration
    very_high = df[df['risk_tier'] == 'VERY_HIGH']
    print(f"\nTop Risk Concentration:")
    print(f"  Top tier (VERY HIGH): {len(very_high):,} patients")
    print(f"  Estimated to represent 25-30% of total care costs")

    # Chronic disease prevalence
    print(f"\nChronic Disease Prevalence:")
    conditions = [
        ('Hypertension', 'has_hypertension'),
        ('Diabetes', 'has_diabetes'),
        ('Heart Disease', 'has_heart_disease'),
        ('COPD', 'has_copd'),
        ('CKD', 'has_ckd'),
        ('Depression', 'has_depression')
    ]
    for name, col in conditions:
        count = df[col].sum()
        pct = df[col].mean() * 100
        print(f"  {name:<18} {count:>6,} ({pct:>5.1f}%)")

    # ── HEDIS Performance ──
    print(f"\n📈 HEDIS QUALITY MEASURE PERFORMANCE")
    print("-"*70)
    print(f"\n{'Measure':<45} {'Rate':>8} {'Natl Avg':>10} {'Target':>8} {'Status':>8}")
    print("-"*82)

    for measure_id, measure in measures.items():
        rate = measure['rate']
        nat_avg = measure['national_average']
        target = measure['target']
        lower_better = measure.get('lower_is_better', False)

        if lower_better:
            if rate <= target:
                status = "✅ PASS"
            elif rate <= nat_avg:
                status = "⚠️  NEAR"
            else:
                status = "❌ FAIL"
        else:
            if rate >= target:
                status = "✅ PASS"
            elif rate >= nat_avg:
                status = "⚠️  NEAR"
            else:
                status = "❌ FAIL"

        name_short = measure['name'][:44]
        print(f"  {name_short:<44} {rate:>7.1f}% {nat_avg:>9.1f}% {target:>7.1f}% {status:>8}")

    # ── Care Gap Summary ──
    print(f"\n🎯 CARE GAP ANALYSIS")
    print("-"*70)

    total_gaps = len(gaps_df)
    unique_patients = gaps_df['patient_token'].nunique()

    print(f"\n  Total open care gaps:     {total_gaps:,}")
    print(f"  Patients with gaps:       {unique_patients:,} "
          f"({unique_patients/len(df)*100:.1f}% of population)")
    print(f"  Average gaps per patient: {total_gaps/unique_patients:.1f}")

    print(f"\n  Gap Priority Distribution:")
    for priority in ['CRITICAL', 'HIGH', 'MODERATE']:
        count = len(gaps_df[gaps_df['priority'] == priority])
        pct = count / total_gaps * 100
        print(f"    {priority:<12} {count:>6,} ({pct:>5.1f}%)")

    print(f"\n  Top 5 Most Common Care Gaps:")
    gap_counts = gaps_df['gap_type'].value_counts().head(5)
    for gap_type, count in gap_counts.items():
        pct = count / unique_patients * 100
        print(f"    {gap_type:<35} {count:>5,} patients ({pct:.1f}%)")

    # ── SDOH Analysis ──
    print(f"\n🏘️  SOCIAL DETERMINANTS OF HEALTH")
    print("-"*70)

    sdoh_items = [
        ('Transportation barriers', 'transportation_barrier'),
        ('Food insecurity', 'food_insecurity'),
        ('Housing instability', 'housing_instability'),
        ('Social isolation', 'social_isolation')
    ]

    for name, col in sdoh_items:
        count = df[col].sum()
        pct = df[col].mean() * 100
        print(f"  {name:<28} {count:>6,} ({pct:>5.1f}%)")

    high_sdoh = df[df['sdoh_risk_score'] >= 2]
    print(f"\n  High SDOH risk (2+ factors): {len(high_sdoh):,} "
          f"({len(high_sdoh)/len(df)*100:.1f}%)")

    # ── Financial Impact ──
    print(f"\n💰 ESTIMATED FINANCIAL IMPACT")
    print("-"*70)

    print(f"\n  HEDIS Star Rating Improvement Opportunity:")

    total_gap_to_target = 0
    for measure_id, measure in measures.items():
        if not measure.get('lower_is_better', False):
            gap = max(0, measure['target'] - measure['rate'])
            if gap > 0:
                total_gap_to_target += gap

    print(f"    Average gap to HEDIS targets: "
          f"{total_gap_to_target/len(measures):.1f} percentage points")

    medicare_pts = len(df[df['insurance'].isin(['Medicare', 'Medicare_Advantage'])])
    star_improvement_value = medicare_pts * 450

    print(f"    Medicare/MA members: {medicare_pts:,}")
    print(f"    Estimated value of 0.5 Star Rating improvement: "
          f"${star_improvement_value:,.0f}/year")

    critical_gaps = len(gaps_df[gaps_df['priority'] == 'CRITICAL'])
    high_gaps = len(gaps_df[gaps_df['priority'] == 'HIGH'])

    avoided_admits = (critical_gaps * 0.30 + high_gaps * 0.15)
    avoided_cost = avoided_admits * 18000

    print(f"\n  Care Gap Closure ROI (if 30% of critical, 15% of high gaps closed):")
    print(f"    Estimated avoided admissions: {avoided_admits:.0f}")
    print(f"    Estimated avoided cost: ${avoided_cost:,.0f}/year")

    # ── Engagement Analytics ──
    print(f"\n📱 PATIENT ENGAGEMENT")
    print("-"*70)

    mychart_rate = df['mychart_enrolled'].mean() * 100
    print(f"\n  MyChart portal enrollment: {mychart_rate:.1f}%")

    contact_pref = df['contact_preference'].value_counts(normalize=True) * 100
    print(f"\n  Preferred contact methods:")
    for method, pct in contact_pref.items():
        print(f"    {method:<12} {pct:.1f}%")

    lost_to_followup = (df['last_pcp_visit_days'] > 365).sum()
    print(f"\n  Lost to follow-up (>12 months no PCP visit): "
          f"{lost_to_followup:,} ({lost_to_followup/len(df)*100:.1f}%)")

    print("\n" + "="*70)
    print("END OF POPULATION HEALTH REPORT")
    print("="*70)


# ─────────────────────────────────────────────
# SECTION 7 — DISEASE-SPECIFIC COHORT ANALYSIS
# ─────────────────────────────────────────────

def analyze_diabetes_cohort(df: pd.DataFrame):
    """
    Deep dive analysis of the diabetic patient cohort.

    This is the kind of analysis a Quality Medical Director
    would request to understand their diabetes population
    and prioritize interventions.
    """

    diabetic = df[df['has_diabetes'] == 1].copy()

    print(f"\n{'='*70}")
    print(f"DIABETES COHORT ANALYSIS")
    print(f"{'='*70}")
    print(f"\nDiabetic patients: {len(diabetic):,}")

    # Control status segmentation
    diabetic['glycemic_control'] = pd.cut(
        diabetic['hba1c'],
        bins=[0, 7.0, 8.0, 9.0, 20],
        labels=['Well Controlled (<7)', 'Controlled (7-8)',
                'Suboptimal (8-9)', 'Poor Control (>9)']
    )

    print(f"\nGlycemic Control Distribution:")
    control_dist = diabetic['glycemic_control'].value_counts()
    for category in ['Well Controlled (<7)', 'Controlled (7-8)',
                     'Suboptimal (8-9)', 'Poor Control (>9)']:
        count = control_dist.get(category, 0)
        pct = count / len(diabetic) * 100
        bar = "█" * int(pct / 3)
        print(f"  {category:<25} {count:>5,} ({pct:>5.1f}%) {bar}")

    # HbA1c distribution statistics
    print(f"\nHbA1c Statistics:")
    print(f"  Mean:   {diabetic['hba1c'].mean():.1f}%")
    print(f"  Median: {diabetic['hba1c'].median():.1f}%")
    print(f"  >8.0%:  {(diabetic['hba1c'] > 8.0).sum():,} patients "
          f"({(diabetic['hba1c'] > 8.0).mean()*100:.1f}%)")
    print(f"  >9.0%:  {(diabetic['hba1c'] > 9.0).sum():,} patients "
          f"({(diabetic['hba1c'] > 9.0).mean()*100:.1f}%)")

    # Comorbidity burden among diabetics
    print(f"\nComorbidities in Diabetic Population:")
    comorbid_items = [
        ('Also have Hypertension', 'has_hypertension'),
        ('Also have Heart Disease', 'has_heart_disease'),
        ('Also have CKD', 'has_ckd'),
        ('Also have Depression', 'has_depression')
    ]
    for name, col in comorbid_items:
        count = diabetic[col].sum()
        pct = diabetic[col].mean() * 100
        print(f"  {name:<28} {count:>5,} ({pct:.1f}%)")

    # Adherence analysis
    print(f"\nMedication Adherence (PDC):")
    print(f"  Mean PDC: {diabetic['diabetes_med_pdc'].mean():.2f}")
    print(f"  Non-adherent (<80% PDC): "
          f"{(diabetic['diabetes_med_pdc'] < 0.80).sum():,} "
          f"({(diabetic['diabetes_med_pdc'] < 0.80).mean()*100:.1f}%)")

    # High-priority subgroup
    critical_diabetics = diabetic[
        (diabetic['hba1c'] > 9.0) &
        (diabetic['days_since_hba1c'] <= 365)
    ]
    print(f"\nCRITICAL SUBGROUP — Poor Control + Tested:")
    print(f"  Count: {len(critical_diabetics):,} patients")
    if len(critical_diabetics) > 0:
        print(f"  Average HbA1c: {critical_diabetics['hba1c'].mean():.1f}%")
        print(f"  Average risk score: {critical_diabetics['risk_score'].mean():.1f}")
    print(f"  On insulin or complex regimen — specialty referral indicated")

    # Intervention targeting
    print(f"\nIntervention Targeting:")
    print(f"  Priority 1 — HbA1c >9% (urgent): "
          f"{(diabetic['hba1c'] > 9.0).sum():,} patients")
    print(f"  Priority 2 — HbA1c 8-9% (suboptimal): "
          f"{((diabetic['hba1c'] > 8.0) & (diabetic['hba1c'] <= 9.0)).sum():,} patients")
    print(f"  Priority 3 — Non-adherent medication: "
          f"{(diabetic['diabetes_med_pdc'] < 0.80).sum():,} patients")
    print(f"  Priority 4 — HbA1c overdue: "
          f"{(diabetic['days_since_hba1c'] > 365).sum():,} patients")


# ─────────────────────────────────────────────
# SECTION 8 — OUTREACH EFFECTIVENESS SIMULATOR
# ─────────────────────────────────────────────

def simulate_outreach_impact(df: pd.DataFrame,
                              gaps_df: pd.DataFrame,
                              measures: dict):
    """
    Simulate the impact of a structured outreach program
    on HEDIS measure performance.

    Shows the ROI of the population health analytics platform
    in terms of quality measure improvement.
    """

    print(f"\n{'='*70}")
    print("OUTREACH IMPACT SIMULATION")
    print("What if we closed 30% of identified care gaps?")
    print("="*70)

    print(f"\nSimulated HEDIS Improvement:")
    print(f"{'Measure':<45} {'Current':>9} {'Projected':>10} {'Gain':>8}")
    print("-"*75)

    for measure_id, measure in measures.items():
        current_rate = measure['rate']
        gap_patients = measure.get('gap_patients', [])
        num_gap_patients = len(gap_patients)
        denominator = measure['denominator']

        if denominator == 0 or num_gap_patients == 0:
            continue

        # Estimate how many gap patients would be closed
        avg_effectiveness = 0.45  # Default outreach effectiveness

        patients_closed = int(num_gap_patients * avg_effectiveness * 0.30)

        # New numerator
        new_numerator = measure['numerator'] + patients_closed
        new_rate = min(new_numerator / denominator * 100, 99.0)

        gain = new_rate - current_rate

        name_short = measure['name'][:44]
        print(f"  {name_short:<44} {current_rate:>8.1f}% {new_rate:>9.1f}% "
              f"+{gain:>5.1f}pp")

    # Financial projection
    print(f"\nFinancial Impact of Outreach Program:")

    medicare_pts = len(df[df['insurance'].isin(['Medicare', 'Medicare_Advantage'])])

    # Star rating improvement
    star_bonus = medicare_pts * 450 * 0.5  # 0.5 star improvement

    # PMPM reduction from better chronic disease management
    diabetic_count = df['has_diabetes'].sum()
    diabetes_cost_reduction = diabetic_count * 0.15 * 12000 * 0.30

    total_impact = star_bonus + diabetes_cost_reduction
    program_cost = 250000  # Annual population health program cost

    print(f"  Star Rating Bonus (0.5 Star improvement): "
          f"${star_bonus:,.0f}/year")
    print(f"  Diabetes Management Savings: "
          f"${diabetes_cost_reduction:,.0f}/year")
    print(f"  Total Financial Impact: ${total_impact:,.0f}/year")
    print(f"  Program Cost: ${program_cost:,.0f}/year")
    print(f"  Net ROI: ${total_impact - program_cost:,.0f}/year")
    print(f"  ROI Ratio: {(total_impact/program_cost):.1f}x")


# ─────────────────────────────────────────────
# SECTION 9 — HEDIS SUMMARY TABLE (for export / dashboard)
# ─────────────────────────────────────────────

def hedis_measures_to_dataframe(measures: dict) -> pd.DataFrame:
    """
    Flatten the HEDIS measures dict into a tidy DataFrame
    (one row per measure) for CSV export and dashboard display.
    Excludes the large per-measure gap_patients lists.
    """
    rows = []
    for measure_id, measure in measures.items():
        lower_better = measure.get('lower_is_better', False)
        rate = measure['rate']
        target = measure['target']
        meets_target = (rate <= target) if lower_better else (rate >= target)
        rows.append({
            'measure_id': measure_id,
            'measure_name': measure['name'],
            'denominator': measure['denominator'],
            'numerator': measure['numerator'],
            'rate': round(rate, 1),
            'national_average': measure['national_average'],
            'target': target,
            'lower_is_better': lower_better,
            'gap_to_target': round(target - rate, 1),
            'meets_target': bool(meets_target),
            'gap_patient_count': len(measure.get('gap_patients', []))
        })
    return pd.DataFrame(rows)
