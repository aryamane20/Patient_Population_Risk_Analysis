"""
Population Health Analytics Platform — Main / Entry Point
Day 19 — Mindbowser Healthcare AI Learning

This file only CALLS the functions defined in population_health.py.
It runs the full analytics pipeline, prints the reports, and exports
the result CSVs / JSON.

Run with:  python3 run_population_health.py
Dashboard: streamlit run population_dashboard.py
"""

import json
from datetime import datetime

from population_health import (
    generate_population,
    calculate_risk_scores,
    calculate_hedis_measures,
    identify_care_gaps,
    generate_daily_worklist,
    generate_population_report,
    analyze_diabetes_cohort,
    simulate_outreach_impact,
    hedis_measures_to_dataframe,
)


def main():

    print("="*70)
    print("POPULATION HEALTH ANALYTICS PLATFORM")
    print("Day 19 — Mindbowser Healthcare AI Learning")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # ── 1. Generate population ──
    df = generate_population(n_patients=10000)

    # ── 2. Risk stratification ──
    print("\nCalculating risk scores...")
    df = calculate_risk_scores(df)

    # ── 3. HEDIS measures ──
    print("Calculating HEDIS measures...")
    measures = calculate_hedis_measures(df)

    # ── 4. Care gap identification ──
    print("Identifying care gaps...")
    print("(This may take a moment for 10,000 patients...)")
    gaps_df = identify_care_gaps(df)
    print(f"Found {len(gaps_df):,} care gaps across "
          f"{gaps_df['patient_token'].nunique():,} patients")

    # ── 5. Daily worklist ──
    print("\nGenerating daily care coordinator worklist...")
    # NOTE: the parameter is `care_coordinator_capacity` (the original
    # call used `capacity=30`, which raised TypeError).
    worklist = generate_daily_worklist(df, gaps_df, care_coordinator_capacity=30)

    print(f"\n{'='*70}")
    print("CARE COORDINATOR DAILY WORKLIST (Top 10)")
    print(f"{'='*70}")
    print(f"\n{'#':<4} {'Token':<12} {'Age':>4} {'Tier':<12} "
          f"{'Score':>6} {'Priority':<10} {'Gaps':>5} {'Channel':<20}")
    print("-"*85)

    for _, row in worklist.head(10).iterrows():
        print(f"  {int(row['outreach_priority']):<3} "
              f"{row['patient_token']:<12} "
              f"{int(row['age']):>4} "
              f"{str(row['risk_tier']):<12} "
              f"{row['risk_score']:>5.0f} "
              f"{row['highest_priority_label']:<10} "
              f"{int(row['num_gaps']):>4} "
              f"{row['suggested_channel']:<20}")

    print(f"\n  Top priority action:")
    top_patient = worklist.iloc[0]
    print(f"  Patient: {top_patient['patient_token']}")
    print(f"  Action: {top_patient['top_action']}")
    print(f"  Contact via: {top_patient['suggested_channel']}")

    # ── 6. Population report ──
    generate_population_report(df, measures, gaps_df)

    # ── 7. Disease cohort analysis ──
    analyze_diabetes_cohort(df)

    # ── 8. Outreach impact simulation ──
    simulate_outreach_impact(df, gaps_df, measures)

    # ── 9. Export ──
    print(f"\n{'='*70}")
    print("EXPORTING DATA")
    print("="*70)

    # Export risk-stratified population
    df.to_csv('population_risk_stratified.csv', index=False)
    print(f"\n✅ Risk-stratified population: population_risk_stratified.csv")

    # Export care gaps
    gaps_df.to_csv('care_gaps.csv', index=False)
    print(f"✅ Care gap list: care_gaps.csv")

    # Export daily worklist
    worklist.to_csv('daily_worklist.csv', index=False)
    print(f"✅ Daily worklist: daily_worklist.csv")

    # Export HEDIS summary (tidy table, one row per measure)
    hedis_df = hedis_measures_to_dataframe(measures)
    hedis_df.to_csv('hedis_measures_summary.csv', index=False)
    print(f"✅ HEDIS summary: hedis_measures_summary.csv")

    # Export a compact JSON snapshot for the dashboard / API consumers
    snapshot = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'population_size': int(len(df)),
        'risk_tier_counts': {
            str(k): int(v) for k, v in df['risk_tier'].value_counts().items()
        },
        'total_care_gaps': int(len(gaps_df)),
        'patients_with_gaps': int(gaps_df['patient_token'].nunique()),
        'hedis_measures': hedis_df.to_dict(orient='records'),
    }
    with open('population_health_summary.json', 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"✅ Summary snapshot: population_health_summary.json")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print("Next: launch the interactive dashboard with")
    print("  streamlit run population_dashboard.py")

    return df, measures, gaps_df, worklist


if __name__ == "__main__":
    main()
