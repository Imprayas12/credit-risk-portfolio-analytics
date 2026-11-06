"""
Credit Risk Portfolio Analytics dashboard.

Run with: streamlit run app/app.py
(Artifacts must already exist — run `python -m src.build` first.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import DIMENSIONS, default_rate_by, cross_default_rate, portfolio_summary, generate_insights
from src.llm_qa import build_grounding_context, answer_question
from src.build import main as run_build

PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
DECISIONS_PATH = ARTIFACT_DIR / "anomaly_decisions.csv"

st.set_page_config(page_title="Credit Risk Portfolio Analytics", layout="wide")


@st.cache_data
def load_data():
    model_df = pd.read_parquet(PROCESSED_DIR / "model_df.parquet")
    customer_df = pd.read_parquet(PROCESSED_DIR / "customer_df.parquet")
    metrics = json.load(open(ARTIFACT_DIR / "metrics.json"))
    feat_imp = pd.read_csv(ARTIFACT_DIR / "feature_importance.csv")
    scored = pd.read_parquet(ARTIFACT_DIR / "scored_accounts.parquet")
    account_scores = pd.read_parquet(ARTIFACT_DIR / "account_scores.parquet")
    dq_log = json.load(open(PROCESSED_DIR / "dq_log.json"))
    return model_df, customer_df, metrics, feat_imp, scored, account_scores, dq_log


artifacts_missing = not (ARTIFACT_DIR / "metrics.json").exists()
if artifacts_missing:
    st.warning("No build artifacts found — running the data/model pipeline once now (this can take ~1 minute).")
    run_build()
    st.cache_data.clear()

model_df, customer_df, metrics, feat_imp, scored, account_scores, dq_log = load_data()
model_df = model_df.merge(account_scores[["account_id", "risk_score", "top_drivers"]], on="account_id", how="left")

# ---------------- Sidebar: global filters + nav ----------------
st.sidebar.title("Filters")
region_f = st.sidebar.multiselect("Region", sorted(model_df["region"].unique()))
loan_f = st.sidebar.multiselect("Loan type", sorted(model_df["loan_type"].unique()))
emp_f = st.sidebar.multiselect("Employment type", sorted(model_df["employment_type"].unique()))

f_df = model_df.copy()
if region_f:
    f_df = f_df[f_df["region"].isin(region_f)]
if loan_f:
    f_df = f_df[f_df["loan_type"].isin(loan_f)]
if emp_f:
    f_df = f_df[f_df["employment_type"].isin(emp_f)]

page = st.sidebar.radio(
    "View",
    ["Portfolio Overview", "Account Explorer", "Ask the Portfolio (AI)", "Anomaly Review (AI)"],
    index=0,
)

st.title("Credit Risk Portfolio Analytics")
st.caption(f"Showing {len(f_df):,} of {len(model_df):,} accounts after filters.")

# ---------------- Portfolio Overview ----------------
if page == "Portfolio Overview":
    summary = portfolio_summary(f_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{summary['total_accounts']:,}")
    c2.metric("Customers", f"{summary['total_customers']:,}")
    c3.metric("Total exposure", f"${summary['total_exposure']:,.0f}")
    c4.metric("Default rate", f"{summary['overall_default_rate']:.1%}")

    st.subheader("Default rate by dimension")
    dim = st.selectbox("Dimension", DIMENSIONS, index=0)
    by_dim = default_rate_by(f_df, dim)
    fig = px.bar(by_dim, x=dim, y="default_rate", hover_data=["accounts", "exposure"],
                 text=by_dim["default_rate"].map(lambda v: f"{v:.1%}"))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-dimensional cut: employment type x loan type")
    cross = cross_default_rate(f_df, "employment_type", "loan_type")
    pivot = cross.pivot(index="employment_type", columns="loan_type", values="default_rate")
    fig2 = px.imshow(pivot, text_auto=".1%", color_continuous_scale="Reds", aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Portfolio insights")
    for ins in generate_insights(f_df, customer_df):
        with st.expander(ins["title"]):
            st.write(ins["detail"])
            st.markdown(f"**So what:** {ins['so_what']}")

    with st.expander("Data-quality decisions (ingestion log)"):
        for table, msgs in dq_log.items():
            st.markdown(f"**{table}**")
            for m in msgs:
                st.write("-", m)

# ---------------- Account Explorer ----------------
elif page == "Account Explorer":
    st.subheader("Account-level risk view")
    acct_id = st.selectbox("Account ID", sorted(f_df["account_id"].unique()))
    row = model_df[model_df["account_id"] == acct_id].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score (P[default])", f"{row['risk_score']:.1%}")
    c2.metric("Actual outcome", "Defaulted" if row["target_default"] == 1 else "Performing")
    c3.metric("Loan type / amount", f"{row['loan_type']} / ${row['loan_amount']:,.0f}")
    st.markdown(f"**Key drivers (SHAP):** {row['top_drivers']}")

    st.markdown("**Repayment behaviour**")
    st.write({
        "months on book": int(row["repay_num_months"]),
        "% missed payments": f"{row['repay_pct_missed']:.1%}",
        "% late payments": f"{row['repay_pct_late']:.1%}",
        "avg days past due": round(row["repay_avg_dpd"], 1),
        "DPD trend (last3 - first3 months)": round(row["repay_dpd_trend"], 1),
    })
    st.markdown("**Bureau snapshot**")
    st.write({
        "external credit lines": int(row["bureau_num_lines"]),
        "utilization": f"{row['bureau_utilization']:.1%}",
        "written-off lines": int(row["bureau_num_written_off"]),
    })

    st.subheader("Global feature importance (XGBoost)")
    st.dataframe(feat_imp.head(15), use_container_width=True)

# ---------------- Ask the Portfolio ----------------
elif page == "Ask the Portfolio (AI)":
    st.subheader("Ask the Portfolio")
    st.caption(
        "Natural-language Q&A grounded in the aggregated portfolio data below — no raw customer rows are sent "
        "to the LLM, and no numbers are invented. If a key isn't set, a deterministic rule-based engine answers "
        "from the same data instead of calling an LLM."
    )
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    import os
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    st.info("LLM mode: ON (OpenAI)" if api_key else "LLM mode: OFF — using rule-based fallback (no API key configured).")

    ctx = build_grounding_context(f_df, metrics=metrics, anomaly_count=int(scored["is_anomaly"].sum()))
    q = st.text_input("Ask a question", placeholder="Which loan type has the highest default rate?")
    if st.button("Ask") and q:
        result = answer_question(q, ctx, api_key=api_key)
        st.success(result["answer"])
    with st.expander("What data is this grounded in?"):
        st.json(ctx)

# ---------------- Anomaly Review ----------------
elif page == "Anomaly Review (AI)":
    st.subheader("Anomaly review queue")
    st.caption(
        "**Trigger:** peer-segment outlier (Isolation Forest, worst 5% within loan_type x employment_type) "
        "or a 30+ day jump in days-past-due vs the start of the loan. "
        "**False positive:** a thin/new account flagged due to an atypical but benign history. "
        "**False negative:** slow deterioration that stays close to an already-risky peer segment average. "
        "**Human-in-the-loop:** analyst confirms (escalate) or dismisses (no action) each flag below."
    )
    flagged = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)

    if DECISIONS_PATH.exists():
        decisions = pd.read_csv(DECISIONS_PATH)
    else:
        decisions = pd.DataFrame(columns=["account_id", "decision", "note"])

    st.write(f"{len(flagged)} accounts flagged.")
    for _, r in flagged.head(30).iterrows():
        with st.expander(f"{r['account_id']} — score {r['anomaly_score']:.2f} — {r['loan_type']}"):
            st.write("Reason:", r["anomaly_reason"])
            existing = decisions[decisions["account_id"] == r["account_id"]]
            default_idx = 0
            if len(existing) and existing.iloc[-1]["decision"] == "Dismiss":
                default_idx = 1
            choice = st.radio("Decision", ["Confirm", "Dismiss"], index=default_idx, key=f"dec_{r['account_id']}", horizontal=True)
            note = st.text_input("Note (optional)", key=f"note_{r['account_id']}")
            if st.button("Save decision", key=f"save_{r['account_id']}"):
                new_row = pd.DataFrame([{"account_id": r["account_id"], "decision": choice, "note": note}])
                decisions = pd.concat([decisions, new_row], ignore_index=True)
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                decisions.to_csv(DECISIONS_PATH, index=False)
                st.success("Saved.")

    if len(decisions):
        st.subheader("Recorded decisions")
        st.dataframe(decisions, use_container_width=True)
