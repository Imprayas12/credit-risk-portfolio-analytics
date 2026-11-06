"""Sanity tests for ingestion, feature engineering, analytics, anomaly
detection and the Ask-the-Portfolio grounding/fallback logic.

Run with: pytest tests/ -q
"""
import pandas as pd
import pytest

from src.ingestion import load_and_clean
from src.features import build_model_df, build_customer_df
from src.analytics import generate_insights, portfolio_summary, default_rate_by
from src.anomaly import detect_anomalies
from src.llm_qa import build_grounding_context, answer_question, DECLINE_MSG


@pytest.fixture(scope="module")
def tables():
    t, _ = load_and_clean()
    return t


@pytest.fixture(scope="module")
def model_df(tables):
    return build_model_df(tables)


@pytest.fixture(scope="module")
def customer_df(tables):
    return build_customer_df(tables)


def test_ingestion_no_missing_key_columns(tables):
    for name, df in tables.items():
        key_col = "customer_id" if "customer_id" in df.columns else df.columns[0]
        assert df[key_col].notna().all(), f"{name}.{key_col} has nulls"


def test_ingestion_gender_normalized(tables):
    assert set(tables["customers"]["gender"].unique()) <= {"Male", "Female"}


def test_model_df_grain_and_no_nulls(model_df, tables):
    assert len(model_df) == tables["loan_accounts"]["account_id"].nunique()
    assert model_df.isna().sum().sum() == 0


def test_model_df_default_rate_reasonable(model_df):
    rate = model_df["target_default"].mean()
    assert 0.05 < rate < 0.35  # dictionary says ~16%


def test_customer_df_grain(customer_df, tables):
    assert len(customer_df) == tables["customers"]["customer_id"].nunique()


def test_insights_have_required_fields(model_df, customer_df):
    insights = generate_insights(model_df, customer_df)
    assert len(insights) >= 4
    for ins in insights:
        assert {"title", "detail", "so_what"} <= ins.keys()
        assert len(ins["so_what"]) > 0


def test_default_rate_by_dimension_sums_to_total_accounts(model_df):
    by_loan = default_rate_by(model_df, "loan_type")
    assert by_loan["accounts"].sum() == len(model_df)


def test_anomaly_detection_flags_reasonable_share(model_df):
    scored = detect_anomalies(model_df)
    share = scored["is_anomaly"].mean()
    assert 0 < share < 0.20  # should flag a minority, not everything


def test_anomaly_reason_present_for_flagged(model_df):
    scored = detect_anomalies(model_df)
    flagged = scored[scored["is_anomaly"]]
    assert (flagged["anomaly_reason"] != "Not flagged").all()


def test_llm_qa_declines_unanswerable_question(model_df):
    ctx = build_grounding_context(model_df)
    result = answer_question("What is the capital of France?", ctx, api_key=None)
    assert result["answer"] == DECLINE_MSG
    assert result["used_llm"] is False


def test_llm_qa_answers_from_real_data(model_df):
    ctx = build_grounding_context(model_df)
    result = answer_question("Which loan type has the highest default rate?", ctx, api_key=None)
    by_loan = default_rate_by(model_df, "loan_type")
    top_loan_type = by_loan.iloc[0]["loan_type"]
    assert top_loan_type in result["answer"]
