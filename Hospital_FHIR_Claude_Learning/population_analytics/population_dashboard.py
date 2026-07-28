"""
Population Health Analytics Platform — Interactive Dashboard
Day 19 — Mindbowser Healthcare AI Learning

A Streamlit dashboard on top of the population_health.py engine.
It runs the analytics live (cached) and lets you explore the
population, HEDIS performance, care gaps, and the care-coordinator
worklist interactively.

Run with:  streamlit run population_dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from population_health import (
    generate_population,
    calculate_risk_scores,
    calculate_hedis_measures,
    identify_care_gaps,
    generate_daily_worklist,
    hedis_measures_to_dataframe,
)

RISK_TIER_ORDER = ['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']

st.set_page_config(
    page_title="Population Health Analytics",
    page_icon="🏥",
    layout="wide",
)


# ─────────────────────────────────────────────
# CACHED ANALYTICS  (recomputes only when inputs change)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Running population health analytics...")
def run_analytics(n_patients: int, seed: int):
    """Run the full engine once and cache the results."""
    df = generate_population(n_patients=n_patients, random_seed=seed)
    df = calculate_risk_scores(df)
    measures = calculate_hedis_measures(df)
    gaps_df = identify_care_gaps(df)
    hedis_df = hedis_measures_to_dataframe(measures)
    return df, measures, gaps_df, hedis_df


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────

st.sidebar.title("⚙️ Controls")
st.sidebar.caption("Population is simulated. Changing these re-runs the engine.")

n_patients = st.sidebar.select_slider(
    "Population size",
    options=[1000, 2500, 5000, 10000, 20000],
    value=5000,
)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
capacity = st.sidebar.slider("Care coordinator daily capacity", 10, 100, 30, step=5)

df, measures, gaps_df, hedis_df = run_analytics(n_patients, int(seed))
worklist = generate_daily_worklist(df, gaps_df, care_coordinator_capacity=capacity)

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
tier_filter = st.sidebar.multiselect(
    "Risk tier",
    options=RISK_TIER_ORDER,
    default=RISK_TIER_ORDER,
)
df_view = df[df['risk_tier'].astype(str).isin(tier_filter)] if tier_filter else df


# ─────────────────────────────────────────────
# HEADER + KPI ROW
# ─────────────────────────────────────────────

st.title("🏥 Population Health Analytics Platform")
st.caption("The analytics engine behind value-based care — risk stratification, "
           "HEDIS measures, and care-gap management.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Attributed patients", f"{len(df):,}")
very_high = int((df['risk_tier'] == 'VERY_HIGH').sum())
k2.metric("Very-high risk", f"{very_high:,}", f"{very_high/len(df)*100:.1f}% of panel")
k3.metric("Open care gaps", f"{len(gaps_df):,}")
patients_with_gaps = gaps_df['patient_token'].nunique() if not gaps_df.empty else 0
k4.metric("Patients with gaps", f"{patients_with_gaps:,}",
          f"{patients_with_gaps/len(df)*100:.1f}%")
measures_passing = int(hedis_df['meets_target'].sum())
k5.metric("HEDIS measures met", f"{measures_passing}/{len(hedis_df)}")

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_overview, tab_hedis, tab_gaps, tab_worklist, tab_diabetes = st.tabs(
    ["📊 Population", "📈 HEDIS", "🎯 Care Gaps", "📋 Worklist", "🩸 Diabetes cohort"]
)


# ── Population overview ──
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk stratification")
        tier_counts = (
            df['risk_tier'].value_counts()
            .reindex(RISK_TIER_ORDER)
            .fillna(0)
            .astype(int)
        )
        st.bar_chart(tier_counts, color="#c0392b")

    with c2:
        st.subheader("Chronic disease prevalence")
        conditions = {
            'Hypertension': 'has_hypertension',
            'Diabetes': 'has_diabetes',
            'Heart Disease': 'has_heart_disease',
            'COPD': 'has_copd',
            'CKD': 'has_ckd',
            'Depression': 'has_depression',
        }
        prev = pd.Series(
            {name: df[col].mean() * 100 for name, col in conditions.items()}
        ).sort_values(ascending=False)
        st.bar_chart(prev, color="#2980b9")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Age distribution")
        age_hist = pd.cut(
            df['age'],
            bins=[18, 30, 45, 55, 65, 75, 95],
            labels=['18-29', '30-44', '45-54', '55-64', '65-74', '75+'],
        ).value_counts().sort_index()
        st.bar_chart(age_hist, color="#16a085")

    with c4:
        st.subheader("SDOH burden")
        sdoh = {
            'Transportation': 'transportation_barrier',
            'Food insecurity': 'food_insecurity',
            'Housing': 'housing_instability',
            'Social isolation': 'social_isolation',
        }
        sdoh_s = pd.Series(
            {name: int(df[col].sum()) for name, col in sdoh.items()}
        )
        st.bar_chart(sdoh_s, color="#8e44ad")

    st.subheader("Care management assignment")
    st.dataframe(
        df['care_management_level'].value_counts().rename_axis('Level').reset_index(name='Patients'),
        use_container_width=True, hide_index=True,
    )


# ── HEDIS ──
with tab_hedis:
    st.subheader("HEDIS quality measure performance")

    show = hedis_df.copy()
    show['status'] = np.where(show['meets_target'], '✅ Meets target', '❌ Below target')

    st.dataframe(
        show[['measure_name', 'denominator', 'numerator', 'rate',
              'national_average', 'target', 'gap_to_target', 'status']]
        .rename(columns={
            'measure_name': 'Measure', 'denominator': 'Denom', 'numerator': 'Numer',
            'rate': 'Rate %', 'national_average': 'Natl avg %', 'target': 'Target %',
            'gap_to_target': 'Gap to target', 'status': 'Status',
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Rate vs. target")
    chart_df = hedis_df.set_index('measure_id')[['rate', 'target', 'national_average']]
    st.bar_chart(chart_df)


# ── Care gaps ──
with tab_gaps:
    if gaps_df.empty:
        st.info("No care gaps identified.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Gaps by priority")
            prio = (
                gaps_df['priority'].value_counts()
                .reindex(['CRITICAL', 'HIGH', 'MODERATE', 'LOW'])
                .dropna().astype(int)
            )
            st.bar_chart(prio, color="#c0392b")
        with c2:
            st.subheader("Most common gap types")
            top_gaps = gaps_df['gap_type'].value_counts().head(8)
            st.bar_chart(top_gaps, color="#d35400")

        st.subheader("Gap explorer")
        prio_pick = st.multiselect(
            "Priority", ['CRITICAL', 'HIGH', 'MODERATE', 'LOW'],
            default=['CRITICAL', 'HIGH'],
        )
        gview = gaps_df[gaps_df['priority'].isin(prio_pick)] if prio_pick else gaps_df
        st.caption(f"{len(gview):,} gaps shown")
        st.dataframe(
            gview[['patient_token', 'age', 'risk_tier', 'risk_score',
                   'gap_type', 'priority', 'description', 'recommended_action']]
            .head(500),
            use_container_width=True, hide_index=True,
        )


# ── Worklist ──
with tab_worklist:
    st.subheader(f"Care coordinator daily worklist (capacity: {capacity})")
    if worklist.empty:
        st.info("No patients on the worklist.")
    else:
        st.dataframe(worklist, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download worklist CSV",
            worklist.to_csv(index=False).encode(),
            file_name="daily_worklist.csv",
            mime="text/csv",
        )


# ── Diabetes cohort ──
with tab_diabetes:
    diabetic = df[df['has_diabetes'] == 1].copy()
    st.subheader(f"Diabetes cohort — {len(diabetic):,} patients")

    if diabetic.empty:
        st.info("No diabetic patients in the current population.")
    else:
        diabetic['glycemic_control'] = pd.cut(
            diabetic['hba1c'],
            bins=[0, 7.0, 8.0, 9.0, 20],
            labels=['Well Controlled (<7)', 'Controlled (7-8)',
                    'Suboptimal (8-9)', 'Poor Control (>9)'],
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Glycemic control**")
            st.bar_chart(diabetic['glycemic_control'].value_counts().sort_index(),
                         color="#c0392b")
        with c2:
            st.markdown("**HbA1c distribution**")
            hist = pd.cut(
                diabetic['hba1c'],
                bins=[4, 6, 7, 8, 9, 10, 11, 15],
            ).value_counts().sort_index()
            hist.index = hist.index.astype(str)
            st.bar_chart(hist, color="#e67e22")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean HbA1c", f"{diabetic['hba1c'].mean():.1f}%")
        m2.metric("HbA1c > 9%", f"{int((diabetic['hba1c'] > 9).sum()):,}")
        m3.metric("Non-adherent (PDC<80%)",
                  f"{int((diabetic['diabetes_med_pdc'] < 0.80).sum()):,}")


st.markdown("---")
st.caption("Simulated data for learning purposes. Engine: population_health.py · "
           "Batch run: run_population_health.py")
