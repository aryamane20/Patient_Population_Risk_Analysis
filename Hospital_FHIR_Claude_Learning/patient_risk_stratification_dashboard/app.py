"""
Patient Risk Stratification Dashboard - Streamlit frontend (Days 26–27).

Role-based views on top of the Day-25 backend artifacts:
  outputs/patient_risk_scores.csv   (per-patient score, tier, top-3 factors)
  data/processed/features.csv       (feature values for per-patient SHAP)
  models/champion_pipeline.joblib   (for on-demand per-patient SHAP)
  models/metrics.json               (LR vs GBM performance)
  outputs/fairness_report.json      (bias monitoring panel)
  outputs/global_importance.csv     (plain-language driver chart)

Audiences (selected in the sidebar):
  - Care Coordinator : Today's Worklist, Patient Explorer, Patient Drill-down
  - Clinical Lead    : Today's Worklist, Population, Bias Monitoring
  - ML Engineer      : everything (adds Decision Threshold, Model Performance)

Run with:  streamlit run app.py   (from the project root, with the .venv active)
"""

from __future__ import annotations

import datetime as dt
import json

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.care_plan import claude_available, generate_care_plan
from src.risk_scoring import percent_of_total

st.set_page_config(page_title="Patient Risk Stratification", layout="wide")

TIER_ORDER = ["High", "Medium", "Low"]
TIER_COLORS = {"High": "#E8684A", "Medium": "#F6BD16", "Low": "#5AD8A6"}


# ─────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading risk scores...")
def load_scores() -> pd.DataFrame:
    return pd.read_csv(config.OUTPUTS_DIR / "patient_risk_scores.csv")


@st.cache_data(show_spinner="Loading features...")
def load_features() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "features.csv")
    return df.set_index(config.PATIENT_ID_COL)


@st.cache_data(show_spinner="Loading outcomes...")
def load_oof_labels():
    """Out-of-fold probabilities + true labels for the threshold tuner."""
    s = load_scores()[["patient_id", "probability"]]
    f = pd.read_csv(config.DATA_PROCESSED / "features.csv")[
        [config.PATIENT_ID_COL, config.TARGET_COL]
    ]
    m = s.merge(f, left_on="patient_id", right_on=config.PATIENT_ID_COL, how="inner")
    return m["probability"].to_numpy(), m[config.TARGET_COL].to_numpy()


@st.cache_data
def load_metrics() -> dict:
    p = config.MODELS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_fairness() -> dict:
    p = config.OUTPUTS_DIR / "fairness_report.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    import joblib
    p = config.MODELS_DIR / "champion_pipeline.joblib"
    return joblib.load(p) if p.exists() else None


def _artifacts_ready() -> bool:
    return (config.OUTPUTS_DIR / "patient_risk_scores.csv").exists()


# ─────────────────────────────────────────────
# Contact log (operational state - deliberately NOT cached, it changes)
# ─────────────────────────────────────────────
def load_contact_log() -> pd.DataFrame:
    """Who was contacted and when. Columns: patient_id, last_contacted (datetime)."""
    p = config.CONTACT_LOG_PATH
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            df["patient_id"] = df["patient_id"].astype(int)
            df["last_contacted"] = pd.to_datetime(df["last_contacted"], errors="coerce")
            return df.dropna(subset=["last_contacted"])
    return pd.DataFrame(columns=["patient_id", "last_contacted"])


def save_contacts(patient_ids, when: dt.date | None = None) -> None:
    """Upsert today's contact date for the given patients (replaces any prior date)."""
    when = when or dt.date.today()
    log = load_contact_log()
    ids = [int(i) for i in patient_ids]
    log = log[~log["patient_id"].isin(ids)]
    new = pd.DataFrame({"patient_id": ids, "last_contacted": pd.Timestamp(when)})
    out = pd.concat([log, new], ignore_index=True)
    out["last_contacted"] = pd.to_datetime(out["last_contacted"]).dt.strftime("%Y-%m-%d")
    out.sort_values("patient_id").to_csv(config.CONTACT_LOG_PATH, index=False)


# ─────────────────────────────────────────────
# Guard: artifacts must exist
# ─────────────────────────────────────────────
if not _artifacts_ready():
    st.title("Patient Risk Stratification Dashboard")
    st.error(
        "No scored patients found. Run the Day-25 backend first:\n\n"
        "```\npython -m src.ingest\npython -m src.clean\npython -m src.features\n"
        "python -m src.model\npython -m src.export\npython -m src.fairness\n```"
    )
    st.stop()

scores = load_scores()
metrics = load_metrics()
fairness = load_fairness()


# ─────────────────────────────────────────────
# Views (one function per tab)
# ─────────────────────────────────────────────
def render_worklist(panel: pd.DataFrame):
    st.subheader("Patients due for follow-up - highest risk first")
    st.markdown(
        "This is your outreach queue. Each patient reappears on a **recurring cadence** "
        "based on their risk tier. Tick **Contacted today** as you reach people and hit "
        "**Save** - they drop off until their next due date, and the next patients move up."
    )
    if len(panel) < len(scores):
        st.info(
            f"Sidebar filters are active - showing {len(panel):,} of {len(scores):,} "
            "patients. Clear the filters to work the full panel."
        )

    with st.expander("Follow-up cadence (days between calls per tier)"):
        cc1, cc2, cc3 = st.columns(3)
        cad = {
            "High": int(cc1.number_input(
                "High risk", 1, 365, config.DEFAULT_CADENCE_DAYS["High"], 1,
                help="How often to call High-risk patients.")),
            "Medium": int(cc2.number_input(
                "Medium risk", 1, 365, config.DEFAULT_CADENCE_DAYS["Medium"], 1)),
            "Low": int(cc3.number_input(
                "Low risk", 1, 365, config.DEFAULT_CADENCE_DAYS["Low"], 1)),
        }

    today = pd.Timestamp(dt.date.today())
    log = load_contact_log()

    wl = panel.merge(log, on="patient_id", how="left")
    # merge against an empty/partial log can yield an object-dtype column; force datetime
    wl["last_contacted"] = pd.to_datetime(wl["last_contacted"], errors="coerce")
    wl["cadence_days"] = wl["tier"].map(cad).fillna(9999).astype(int)
    wl["next_due"] = wl["last_contacted"] + pd.to_timedelta(wl["cadence_days"], unit="D")
    never = wl["last_contacted"].isna()
    wl["status"] = np.select(
        [never, wl["next_due"] < today, wl["next_due"] <= today],
        ["New", "Overdue", "Due"], default="Scheduled",
    )
    due_mask = never | (wl["next_due"] <= today)
    due = wl[due_mask].copy()

    n_due = len(due)
    n_over = int((due["status"] == "Overdue").sum())
    n_new = int((due["status"] == "New").sum())
    n_sched = int((~due_mask).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Due today", f"{n_due:,}")
    c2.metric("Overdue", f"{n_over:,}", help="Past their next-due date.")
    c3.metric("Never contacted", f"{n_new:,}")
    c4.metric("Scheduled (not yet due)", f"{n_sched:,}",
              help="Already contacted; will return when their cadence elapses.")

    # Status first (Overdue > New > Due), then risk within each status.
    status_rank = {"Overdue": 0, "New": 1, "Due": 2}
    due["_s"] = due["status"].map(status_rank).fillna(3)
    due = due.sort_values(["_s", "risk_score"], ascending=[True, False])

    cap_col, _ = st.columns([2, 3])
    cap_on = cap_col.checkbox("Only show my first N patients", value=True)
    if cap_on:
        cap = int(cap_col.number_input("How many to work now?", 5, 1000, 30, 5))
        disp = due.head(cap)
    else:
        disp = due

    if disp.empty:
        st.success("Nothing due right now - every patient is within their cadence window.")
        return

    editor_df = pd.DataFrame({
        "contacted_today": False,
        "patient_id": disp["patient_id"].to_numpy(),
        "risk_score": disp["risk_score"].to_numpy(),
        "tier": disp["tier"].to_numpy(),
        "status": disp["status"].to_numpy(),
        "last_contacted": disp["last_contacted"].dt.date,
        "next_due": disp["next_due"].dt.date,
        "why": disp["top_3_risk_factors"].to_numpy(),
    })

    edited = st.data_editor(
        editor_df, width="stretch", hide_index=True,
        key=f"worklist_editor_{len(log)}",
        column_config={
            "contacted_today": st.column_config.CheckboxColumn(
                "Contacted today", help="Tick after you reach the patient, then Save."),
            "patient_id": st.column_config.NumberColumn("Patient ID"),
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%d"),
            "tier": st.column_config.TextColumn("Tier"),
            "status": st.column_config.TextColumn(
                "Status", help="New = never contacted · Overdue = past due · Due = due today."),
            "last_contacted": st.column_config.DateColumn("Last contacted"),
            "next_due": st.column_config.DateColumn("Next due"),
            "why": st.column_config.TextColumn(
                "Why to call (top risk factors)", width="large"),
        },
        disabled=["patient_id", "risk_score", "tier", "status",
                  "last_contacted", "next_due", "why"],
    )

    b1, b2 = st.columns([1, 3])
    if b1.button("Save contacts", type="primary"):
        ids = edited.loc[edited["contacted_today"], "patient_id"].tolist()
        if ids:
            save_contacts(ids)
            st.success(f"Logged {len(ids)} contact(s) for {dt.date.today():%b %d, %Y}.")
            st.rerun()
        else:
            st.info("No patients ticked yet - tick 'Contacted today' first.")
    b2.download_button(
        "Download this worklist (CSV)",
        editor_df.drop(columns="contacted_today").to_csv(index=False).encode(),
        file_name="todays_worklist.csv", mime="text/csv",
    )


def render_population():
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Patients by risk tier")
        tier_counts = scores["tier"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
        tier_df = pd.DataFrame({
            "tier": tier_counts.index,
            "patients": tier_counts.values,
            "pct_of_panel": percent_of_total(tier_counts.values),
        })
        tier_chart = (
            alt.Chart(tier_df).mark_bar().encode(
                x=alt.X("tier:N", sort=TIER_ORDER, title="Risk tier"),
                y=alt.Y("patients:Q", title="Number of patients"),
                color=alt.Color("tier:N",
                                scale=alt.Scale(domain=TIER_ORDER,
                                                range=[TIER_COLORS[t] for t in TIER_ORDER]),
                                legend=None),
                tooltip=[alt.Tooltip("tier:N", title="Risk tier"),
                         alt.Tooltip("patients:Q", title="Patients", format=","),
                         alt.Tooltip("pct_of_panel:Q", title="% of panel", format=".1f")],
            ).properties(width="container", height=340)
        )
        st.altair_chart(tier_chart)
        st.caption("Hover a bar to see the patient count and its share of the panel.")
    with c2:
        st.subheader("Risk score distribution")
        edges = list(range(0, 105, 5))
        cut = pd.cut(scores["risk_score"], bins=edges, right=False, include_lowest=True)
        hist = cut.value_counts().sort_index()
        hist_df = pd.DataFrame({
            "band": [f"{int(iv.left)}-{int(iv.right)}" for iv in hist.index],
            "lo": [int(iv.left) for iv in hist.index],
            "patients": hist.values,
        })
        hist_df["pct_of_panel"] = percent_of_total(hist_df["patients"])
        score_chart = (
            alt.Chart(hist_df).mark_bar(color="#5B8FF9").encode(
                x=alt.X("band:N", sort=list(hist_df.sort_values("lo")["band"]),
                        title="Risk score band (0 = low risk, 100 = high risk)"),
                y=alt.Y("patients:Q", title="Number of patients"),
                tooltip=[alt.Tooltip("band:N", title="Score band"),
                         alt.Tooltip("patients:Q", title="Patients", format=","),
                         alt.Tooltip("pct_of_panel:Q", title="% of panel", format=".1f")],
            ).properties(width="container", height=340)
        )
        st.altair_chart(score_chart)
        st.caption("Each bar counts patients in a 5-point band. Hover for exact counts.")

    st.subheader("Tier summary")
    summ = (
        scores.groupby("tier")
        .agg(patients=("risk_score", "size"),
             mean_score=("risk_score", "mean"),
             mean_probability=("probability", "mean"))
        .reindex(TIER_ORDER)
    )
    summ["mean_score"] = summ["mean_score"].round(1)
    summ["mean_probability"] = summ["mean_probability"].round(4)
    summ["pct_of_panel"] = percent_of_total(summ["patients"])
    st.dataframe(summ.reset_index(), width="stretch", hide_index=True)


def render_explorer(view: pd.DataFrame):
    st.subheader(f"Risk-patient list - {len(view):,} patients (filtered)")
    st.caption("Use the sidebar to filter by tier and score; click a column to sort.")
    st.dataframe(
        view[["patient_id", "risk_score", "tier", "top_3_risk_factors",
              "probability", "predicted_label"]].head(1000),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download filtered list (CSV)",
        view.to_csv(index=False).encode(),
        file_name="filtered_risk_scores.csv", mime="text/csv",
    )


def render_drilldown(view: pd.DataFrame):
    st.subheader("Individual patient")
    default_list = (view if len(view) else scores).sort_values(
        "risk_score", ascending=False)["patient_id"].tolist()
    pid = st.selectbox(
        "Select patient (filtered, highest risk first)",
        options=default_list[:500], format_func=lambda x: f"Patient {x}",
    )
    row = scores[scores["patient_id"] == pid].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", f"{int(row['risk_score'])}/100")
    m2.metric("Tier", row["tier"])
    m3.metric("Model probability", f"{row['probability']:.3f}")

    st.markdown(f"**Top risk factors:** {row['top_3_risk_factors']}")

    st.markdown("#### Why this patient? (risk factor breakdown)")
    try:
        from src.explain import patient_contributions
        from src.features import FEATURE_SCHEMA

        feats = load_features()
        pipe = load_pipeline()
        if pipe is not None and pid in feats.index:
            X_row = feats.loc[[pid], FEATURE_SCHEMA.all_features]
            contrib = patient_contributions(pipe, X_row, top_n=8)
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#E8684A" if v > 0 else "#5AD8A6" for v in contrib["shap_value"]]
            ax.barh(contrib["label"], contrib["shap_value"], color=colors)
            ax.invert_yaxis()
            ax.axvline(0, color="#444", linewidth=0.8)
            ax.set_xlabel("Effect on risk  (red = raises risk, green = lowers risk)")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Per-patient breakdown unavailable (model or feature row missing).")
    except Exception as exc:
        st.warning(f"Could not compute the breakdown for this patient: {exc}")

    st.markdown("#### Suggested care plan")
    engine = "Claude" if claude_available() else "rule-based engine"
    if st.button(f"Generate care plan ({engine})", type="primary"):
        with st.spinner("Generating care plan..."):
            plan = generate_care_plan(row.to_dict())
        st.caption(f"Source: {'Claude' if plan['source'] == 'claude' else 'Rule-based'}")
        st.info(plan["summary"])
        for iv in plan["interventions"]:
            pr = iv.get("priority", "")
            st.markdown(f"**[{pr}]** **{iv.get('category','')}** - {iv.get('action','')}")


def render_threshold():
    from src.threshold import recommended_threshold, sweep_thresholds

    st.subheader("How many patients do we flag? (coverage vs. workload dial)")
    st.markdown(
        "Where you draw the 'flag for outreach' line is a choice: a lower line "
        "**catches more of the patients who will actually return** but creates **more "
        "false alarms**. A missed readmission usually costs far more than a phone call, "
        "so most programs flag more, not fewer."
    )
    proba, y_true = load_oof_labels()

    @st.cache_data(show_spinner=False)
    def _sweep(_key: int):
        return sweep_thresholds(proba, y_true)
    sweep = _sweep(len(proba))

    cost_ratio = st.slider(
        "How many unnecessary outreach calls are you willing to make to catch ONE more readmission?",
        min_value=1, max_value=40, value=10, step=1,
        help="Cost of a missed readmission relative to one follow-up call. "
             "Higher = flag more patients.",
    )
    thr, best, sweep = recommended_threshold(proba, y_true, cost_fn=float(cost_ratio),
                                             cost_fp=1.0, sweep=sweep)
    default_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]
    total_pos = int(best["tp"] + best["fn"])

    st.markdown(f"#### Recommended: flag patients scoring above **{thr*100:.0f}/100**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readmissions caught", f"{best['recall']*100:.0f}%",
              f"{(best['recall']-default_row['recall'])*100:+.0f} pts vs default",
              help="Share of patients who WILL be readmitted that we flag.")
    c2.metric("Patients flagged", f"{best['flagged_pct']:.0f}% of panel",
              help="How much outreach workload this creates.")
    c3.metric("Caught / Missed", f"{int(best['tp']):,} / {int(best['fn']):,}",
              help=f"Out of {total_pos:,} patients who were actually readmitted.")
    c4.metric("Flag accuracy", f"{(best['precision'] or 0)*100:.0f}%",
              help="Of flagged patients, the share actually readmitted.")

    st.caption(
        f"For comparison, the default 0.5 cut-off catches {default_row['recall']*100:.0f}% "
        f"of readmissions while flagging {default_row['flagged_pct']:.0f}% of the panel."
    )

    pr = sweep.dropna(subset=["precision"])
    base = alt.Chart(pr).encode(
        x=alt.X("recall:Q", title="Readmissions caught (recall)", axis=alt.Axis(format="%")),
        y=alt.Y("precision:Q", title="Flag accuracy (precision)", axis=alt.Axis(format="%")),
    )
    line = base.mark_line(color="#5B8FF9")
    point = (
        alt.Chart(pd.DataFrame([{"recall": best["recall"], "precision": best["precision"]}]))
        .mark_point(size=140, color="#E8684A", filled=True)
        .encode(x="recall:Q", y="precision:Q",
                tooltip=[alt.Tooltip("recall:Q", title="Caught", format=".0%"),
                         alt.Tooltip("precision:Q", title="Accuracy", format=".0%")])
    )
    st.altair_chart((line + point).properties(width="container", height=340))
    st.caption("Blue line: every possible operating point. Red dot: your current choice.")

    with st.expander("Why isn't the model just 'more accurate' instead?"):
        st.markdown(
            "- Ranking quality tops out ~0.65 on this administrative dataset - a known "
            "literature ceiling. More rows of the same data won't move it.\n"
            "- The biggest gains would come from **richer data per patient** (vitals, lab "
            "trends, medication adherence, social factors, discharge notes).\n"
            "- Until then this dial is the main lever: set how wide to cast the net."
        )


def render_model():
    st.subheader("How good is the model?")
    champ = metrics.get("models", {}).get(metrics.get("champion", "gbm"), {})
    if champ:
        auroc = champ.get("auroc", 0)
        recall = champ.get("recall", 0)
        cm = champ.get("confusion_matrix", {})
        st.success(
            "**In plain language:** if you pick one patient who *was* readmitted and one "
            f"who *wasn't*, the model ranks the readmitted one higher about "
            f"**{auroc*100:.0f}% of the time**, and at the current setting catches about "
            f"**{recall*100:.0f}% of eventual readmissions** - a prioritization aid, not a "
            "perfect predictor."
        )
        friendly = {
            "auroc": "Ranking quality", "pr_auc": "High-risk flag quality",
            "recall": "% of readmissions caught", "precision": "% of flags that were right",
            "f1": "Overall balance", "brier": "Probability error (lower=better)",
        }
        helps = {
            "auroc": "Chance the model ranks a truly readmitted patient above a non-readmitted one. 0.5 = coin flip.",
            "pr_auc": "Quality of the high-risk flags given only ~9% actually readmit.",
            "recall": "Of patients who were readmitted, the share the model flagged.",
            "precision": "Of patients the model flags, the share actually readmitted.",
            "f1": "A single number balancing recall and precision.",
            "brier": "Average error of the predicted probabilities. Lower is better.",
        }
        comp = pd.DataFrame(metrics["models"]).T[list(friendly)]
        comp.index = comp.index.str.upper()
        comp = comp.rename(columns=friendly).reset_index(names="Model")
        st.dataframe(
            comp, width="stretch", hide_index=True,
            column_config={friendly[k]: st.column_config.NumberColumn(
                friendly[k], help=helps[k], format="%.3f") for k in friendly},
        )
        st.caption(
            f"Champion: **{metrics.get('champion','?').upper()}**. Cohort "
            f"{metrics.get('n_rows',0):,} patients, {metrics.get('positive_rate',0)*100:.1f}% "
            "readmitted. ~0.65 ranking quality matches published results - reported honestly."
        )
        if cm:
            st.markdown("**What happened on the test patients (at the default setting):**")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Correctly flagged", f"{cm.get('tp',0):,}",
                      help="Readmitted patients the model flagged.")
            q2.metric("Missed", f"{cm.get('fn',0):,}",
                      help="Readmitted but not flagged - the costly miss.")
            q3.metric("False alarms", f"{cm.get('fp',0):,}",
                      help="Flagged but didn't return - extra outreach.")
            q4.metric("Correctly cleared", f"{cm.get('tn',0):,}",
                      help="Low-risk patients correctly not flagged.")

    st.subheader("What drives readmission risk across all patients?")
    gi_path = config.OUTPUTS_DIR / "global_importance.csv"
    if gi_path.exists():
        gi = pd.read_csv(gi_path)
        st.markdown(
            "Longer bar = bigger influence on the model's risk scores across the panel "
            "(as a share of the model's total influence)."
        )
        drivers_chart = (
            alt.Chart(gi).mark_bar(color="#5B8FF9").encode(
                x=alt.X("relative_influence:Q", title="Relative influence on risk (%)"),
                y=alt.Y("label:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("label:N", title="Factor"),
                         alt.Tooltip("relative_influence:Q", title="Relative influence",
                                     format=".1f")],
            ).properties(width="container", height=380)
        )
        st.altair_chart(drivers_chart)
        st.info(f"**Biggest drivers:** {', '.join(gi['label'].head(3))}.")
        with st.expander("How do we know what drives risk?"):
            st.markdown(
                "For each patient we measure how much every factor pushed their risk up or "
                "down (**SHAP**), then average across all patients. The Patient Drill-down "
                "shows the same breakdown for one patient."
            )
    else:
        st.caption("Run `python -m src.export` to generate the driver breakdown.")


def render_bias():
    st.subheader("Is the model fair across patient groups?")
    st.markdown(
        "We check whether the model works **equally well for every group of patients**. "
        "The key column is **'High-risk missed'** - the share of patients who were "
        "readmitted but the model did *not* flag. Big differences between groups mean the "
        "tool is quietly under-serving some patients."
    )
    with st.expander("What do these columns mean? (plain language)"):
        st.markdown(
            "- **Patients** - how many patients are in this group.\n"
            "- **Actual readmit rate** - how often this group truly returned within 30 days.\n"
            "- **Ranking quality** - how well the model orders risk within the group.\n"
            "- **High-risk caught** - of those who returned, the share the model flagged.\n"
            "- **High-risk missed** - of those who returned, the share the model MISSED "
            "(lower is better).\n"
            "- **Flagged rate** - share of the group the model marks high-risk.\n\n"
            "**Gaps** summarize the spread between groups. Closer to 0 = more equal."
        )
    friendly = {
        "subgroup": "Group", "n": "Patients", "positive_rate": "Actual readmit rate",
        "auroc": "Ranking quality", "recall_tpr": "High-risk caught",
        "fnr": "High-risk missed", "selection_rate": "Flagged rate",
    }
    pct_cols = {"positive_rate", "recall_tpr", "fnr", "selection_rate"}
    col_help = {
        "n": "Number of patients in this group.",
        "positive_rate": "How often this group truly returned within 30 days.",
        "auroc": "Ranking quality within the group (0.5 = coin flip, 1.0 = perfect).",
        "recall_tpr": "Of those who returned, the share the model flagged.",
        "fnr": "Of those who returned, the share the model MISSED (lower is better).",
        "selection_rate": "Share of the group the model marks high-risk.",
    }
    attr_titles = {"race": "Race", "gender": "Gender", "age_band": "Age band"}
    if not fairness.get("tables"):
        st.info("No fairness report found. Run `python -m src.fairness`.")
        return
    for attr, records in fairness["tables"].items():
        st.markdown(f"### By {attr_titles.get(attr, attr)}")
        tbl = pd.DataFrame(records)
        tbl = tbl[[c for c in friendly if c in tbl.columns]].rename(columns=friendly)
        st.dataframe(
            tbl, width="stretch", hide_index=True,
            column_config={friendly[c]: st.column_config.NumberColumn(
                friendly[c], help=col_help.get(c, ""),
                format="percent" if c in pct_cols else None)
                for c in friendly if c != "subgroup"},
        )
        gaps = fairness.get("gaps", {}).get(attr, {})
        if gaps:
            worst = gaps.get("worst_fnr_subgroup", "-")
            worst_v = gaps.get("worst_fnr")
            if worst and worst_v is not None:
                st.warning(
                    f"**Plain-language read:** the model misses the most readmissions in the "
                    f"**{worst}** group - about **{worst_v*100:.0f}%** go unflagged. Worth a "
                    "manual review step there."
                )
            g1, g2 = st.columns(2)
            g1.metric("Equal-opportunity gap", gaps.get("equal_opportunity_diff", "-"),
                      help="Spread in 'High-risk caught' across groups. 0 = equal.")
            g2.metric("Flagging (parity) gap", gaps.get("demographic_parity_diff", "-"),
                      help="Spread in how often each group is flagged. 0 = equal.")


# ─────────────────────────────────────────────
# Sidebar: role selector + filters
# ─────────────────────────────────────────────
st.sidebar.title("View")
role = st.sidebar.radio(
    "I am a…",
    ["Care Coordinator", "Clinical Lead", "ML Engineer"],
    help="Switches the dashboard to show only what's relevant to your role.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
st.sidebar.caption("Apply to Today's Worklist, Patient Explorer & Drill-down.")
tier_filter = st.sidebar.multiselect("Risk tier", TIER_ORDER, default=TIER_ORDER)
score_min, score_max = st.sidebar.slider("Risk score range", 0, 100, (0, 100))
view = scores[
    scores["tier"].isin(tier_filter) & scores["risk_score"].between(score_min, score_max)
]

st.sidebar.markdown("---")
if claude_available():
    st.sidebar.success("Claude API detected - AI care plans enabled.")
else:
    st.sidebar.info("No ANTHROPIC_API_KEY - care plans use the rule-based engine.")


# ─────────────────────────────────────────────
# Header + role-appropriate KPI row
# ─────────────────────────────────────────────
st.title("Patient Risk Stratification Dashboard")

n = len(scores)
high = int((scores["tier"] == "High").sum())
if role == "ML Engineer":
    st.caption("30-day readmission risk - full analytical view.")
    gbm = metrics.get("models", {}).get(metrics.get("champion", "gbm"), {})
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Patients scored", f"{n:,}")
    k2.metric("High risk", f"{high:,}", f"{high/n*100:.1f}% of panel")
    k3.metric("Mean risk score", f"{scores['risk_score'].mean():.1f}")
    k4.metric("Model ranking quality", f"{gbm.get('auroc', float('nan')):.3f}")
    k5.metric("Readmission base rate", f"{metrics.get('positive_rate', 0)*100:.1f}%")
else:
    st.caption("Who to follow up after discharge, and why.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Patients in panel", f"{n:,}")
    k2.metric("High-risk patients", f"{high:,}", f"{high/n*100:.1f}% of panel")
    k3.metric("Mean risk score", f"{scores['risk_score'].mean():.1f}")

st.markdown("---")


# ─────────────────────────────────────────────
# Role-based tabs
# ─────────────────────────────────────────────
ROLE_TABS = {
    "Care Coordinator": [
        ("Today's Worklist", lambda: render_worklist(view)),
        ("Patient Explorer", lambda: render_explorer(view)),
        ("Patient Drill-down", lambda: render_drilldown(view)),
    ],
    "Clinical Lead": [
        ("Today's Worklist", lambda: render_worklist(view)),
        ("Population", lambda: render_population()),
        ("Bias Monitoring", lambda: render_bias()),
    ],
    "ML Engineer": [
        ("Population", lambda: render_population()),
        ("Patient Explorer", lambda: render_explorer(view)),
        ("Patient Drill-down", lambda: render_drilldown(view)),
        ("Decision Threshold", lambda: render_threshold()),
        ("Model Performance", lambda: render_model()),
        ("Bias Monitoring", lambda: render_bias()),
    ],
}

specs = ROLE_TABS[role]
tabs = st.tabs([label for label, _ in specs])
for tab, (_, render) in zip(tabs, specs):
    with tab:
        render()


st.markdown("---")
st.caption(
    "Backend: src/ pipeline (Day 25) · Dashboard: Day 26 · AI care plans: Day 27. "
    "Data: UCI Diabetes 130-US Hospitals (CC BY 4.0)."
)
