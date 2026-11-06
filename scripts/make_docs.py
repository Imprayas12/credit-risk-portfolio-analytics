"""One-off script (not part of the app) to render the two required
one-page PDFs: docs/architecture.pdf and docs/solution_overview.pdf.
Run: python scripts/make_docs.py
Requires `reportlab` (doc-generation only, not an app runtime dependency).
"""
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=3, spaceBefore=8)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=8.7, leading=11)


def doc(path, margin=0.5 * inch):
    return SimpleDocTemplate(str(path), pagesize=LETTER,
                              leftMargin=margin, rightMargin=margin,
                              topMargin=margin, bottomMargin=margin)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=6) for i in items],
                         bulletType="bullet", start="•", leftIndent=12, spaceAfter=3)


# ---------------- architecture.pdf ----------------
story = [Paragraph("Credit Risk Portfolio Analytics — Architecture", H1)]

story.append(Paragraph("Data flow: raw CSVs → ingestion → analytics → model → dashboard", H2))
story.append(Table(
    [["Stage", "Tool / library", "Output"],
     ["1. Raw data", "5 CSVs in data/raw/ (pandas.read_csv)", "customers, loan_accounts, bureau_credit,\nrepayment_history, collections_events"],
     ["2. Ingestion & cleaning", "src/ingestion.py (pandas)", "gender normalized, missing values imputed\n+ flagged, referential-integrity checks logged"],
     ["3. Feature engineering", "src/features.py (pandas groupby/agg)", "model_df (account grain) & customer_df\n(customer grain) — joins bureau/repayment/\ncollections rollups onto loans/customers"],
     ["4. Analytics", "src/analytics.py", "default-rate cuts by dimension, cross-tabs,\nauto-generated insights"],
     ["5. Model", "src/model.py (scikit-learn LogisticRegression,\nXGBoost, SHAP)", "AUC/KS/Gini, feature importance, SHAP\ndrivers per account → artifacts/*.joblib, *.csv"],
     ["6. Anomaly detection", "src/anomaly.py (IsolationForest per peer\nsegment + DPD-jump rule)", "scored_accounts.parquet with anomaly_score,\nflag reasons"],
     ["7. Ask-the-Portfolio (LLM)", "src/llm_qa.py (OpenAI chat completions,\nrule-based fallback)", "grounded Q&A answers"],
     ["8. Dashboard", "app/app.py (Streamlit, Plotly)", "Portfolio view, Account view, Ask-the-\nPortfolio, Anomaly review"]],
    colWidths=[1.3 * inch, 2.5 * inch, 3.2 * inch],
    style=TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]),
))

story.append(Paragraph("Where the LLM is called & what data is sent", H2))
story.append(bullets([
    "The LLM is called only inside src/llm_qa.answer_question(), invoked from the \"Ask the Portfolio (AI)\" "
    "page in app/app.py, one request per analyst question.",
    "What is sent: a small pre-aggregated JSON context (default-rate tables by dimension and cross-tab, "
    "model metrics, anomaly count) built by build_grounding_context() — never raw customer/account rows, "
    "names, or PII.",
    "The system prompt instructs the model to answer strictly from that JSON, quote exact figures, and "
    "decline if the question can't be answered from it.",
    "If OPENAI_API_KEY is not configured, the same grounded JSON is answered by a deterministic rule-based "
    "matcher (src/llm_qa._rule_based_answer) instead of an LLM call — the feature still works, with no risk "
    "of hallucination either way.",
]))

story.append(Paragraph("Both AI features embedded in the dashboard", H2))
story.append(bullets([
    "Ask the Portfolio: text box + answer on its own dashboard page, grounded in live-filtered data.",
    "Anomaly detection: review queue page listing flagged accounts with reasons, plus an analyst "
    "Confirm/Dismiss control that is persisted to artifacts/anomaly_decisions.csv (human-in-the-loop).",
]))

doc(DOCS / "architecture.pdf").build(story)

# ---------------- solution_overview.pdf ----------------
story2 = [Paragraph("Credit Risk Portfolio Analytics — Solution Overview", H1)]

story2.append(Paragraph("Dashboard, view by view", H2))
story2.append(bullets([
    "Portfolio Overview: exposure, accounts, overall default rate; default rate by any single dimension "
    "(loan type, employment, region, income band, age band); an employment-type x loan-type heatmap "
    "(cross-dimensional cut); 5 auto-generated insights with a \"so what\"; the data-quality decision log. "
    "An analyst uses this to spot which segments are driving losses and where to tighten policy.",
    "Account Explorer: pick any account_id (respecting the sidebar filters) to see its risk score, actual "
    "outcome, top SHAP drivers, repayment behaviour and bureau snapshot, plus global feature importance. "
    "An analyst uses this to explain a specific account's risk score to a credit committee.",
    "Ask the Portfolio (AI): free-text question box grounded in the currently filtered data.",
    "Anomaly Review (AI): ranked queue of flagged accounts with the reason for each flag and a "
    "Confirm/Dismiss control an analyst uses during periodic portfolio reviews.",
]))

story2.append(Paragraph("Two AI features", H2))
story2.append(bullets([
    "Ask the Portfolio: LLM (OpenAI, falls back to rule-based matching with no key) answers from a small "
    "pre-aggregated JSON context — never raw rows — and declines when the answer isn't in that context.",
    "Anomaly detection — trigger: Isolation Forest outlier (worst 5%) within an account's loan_type x "
    "employment_type peer segment on repayment/bureau behaviour features, OR a 30+ day DPD jump versus the "
    "start of the loan. False positive: a thin/new account with an atypical but benign history gets "
    "flagged. False negative: slow deterioration that stays near an already-risky segment's average is "
    "missed. Human-in-the-loop: analyst confirms (escalate) or dismisses each flag in the review queue; "
    "decisions are logged to artifacts/anomaly_decisions.csv.",
]))

story2.append(Paragraph("Model results & approach", H2))
story2.append(Paragraph(
    "Target: target_default on the account grain (14,685 accounts, ~16.1% base rate). Features: customer "
    "demographics, loan terms, bureau rollups (utilization, DPD, write-offs) and repayment-history rollups "
    "(missed/late %, DPD trend, underpayment) aggregated per account. Baseline: Logistic Regression "
    "(class-balanced) — AUC 0.662, KS 0.244, Gini 0.323. Challenger: XGBoost — AUC 0.643, KS 0.207, Gini "
    "0.286 (comparable discrimination, adds non-linear interactions and SHAP-based per-account explanations; "
    "LR is the more stable/interpretable production baseline here). Top drivers: loan_type=Mortgage, "
    "repay_avg_dpd, annual_income, repay_pct_late, employment_type=Unemployed.", BODY))
story2.append(Paragraph("Monitoring after deployment: track population stability index (PSI) on inputs and "
    "score distribution monthly, recalculate AUC/KS/Gini on each new vintage once defaults mature, alert on "
    "feature drift (e.g. bureau_utilization, repay_pct_missed), and retrain on a fixed cadence or when PSI "
    "exceeds a threshold.", BODY))

story2.append(Paragraph("Top 3 portfolio insights", H2))
story2.append(bullets([
    "Unemployed borrowers default at ~35% vs ~13.6% for Business Owners — employment type should drive "
    "underwriting/pricing, not just rate.",
    "Mortgage is the riskiest product at ~24.7% default despite being secured, and carries the largest "
    "dollar exposure — pricing/collateral terms deserve review.",
    "Unemployed + Mortgage is a specific hotspot at ~54% default (n=70) — this combination needs its own "
    "rule rather than being averaged into broader policies.",
]))

story2.append(Paragraph("Key assumptions & production needs", H2))
story2.append(bullets([
    "Assumptions: synthetic data's missingness/labels mirror real messiness; a 2026-01-01 reference date for "
    "age-of-loan calcs; peer segments with <20 accounts are too small for stable anomaly scoring.",
    "From IT/data teams: a refreshed daily/weekly data feed (not static CSVs), a feature store or warehouse "
    "view for the bureau/repayment rollups, secrets management for the LLM key in production, and a model "
    "registry + approval workflow before any score is used for decisions.",
]))

doc(DOCS / "solution_overview.pdf").build(story2)
print("Wrote", DOCS / "architecture.pdf", "and", DOCS / "solution_overview.pdf")
