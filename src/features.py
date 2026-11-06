"""
Feature engineering: turns the one-to-many tables (bureau, repayment,
collections) into account-level and customer-level aggregates, then joins
everything into two analysis-ready datasets:

- `model_df`  : one row per loan account (grain of target_default) — used
                for the default model and the account-level dashboard view.
- `customer_df`: one row per customer — used for portfolio profiling
                (income x employment x region x age x loan type).

Join logic
----------
loan_accounts (account grain, holds target)
  + customers            on customer_id            (many accounts -> 1 customer)
  + bureau_agg            on customer_id            (external credit, many -> 1 customer)
  + repayment_agg          on account_id             (monthly history -> 1 account)
  + collections_agg        on account_id             (collections events -> 1 account)

customer_df = customers + loan_agg (per-customer rollup of loan_accounts)
              + bureau_agg
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    g = bureau.groupby("customer_id")
    out = g.agg(
        bureau_num_lines=("bureau_id", "count"),
        bureau_total_sanctioned=("sanctioned_amount", "sum"),
        bureau_total_outstanding=("outstanding_amount", "sum"),
        bureau_avg_dpd=("days_past_due", "mean"),
        bureau_max_dpd=("days_past_due", "max"),
        bureau_num_written_off=("is_written_off", "sum"),
        bureau_num_active=("account_status", lambda s: (s == "Active").sum()),
    ).reset_index()
    out["bureau_utilization"] = (
        out["bureau_total_outstanding"] / out["bureau_total_sanctioned"].replace(0, np.nan)
    ).fillna(0).clip(0, 5)
    return out


def repayment_features(repay: pd.DataFrame) -> pd.DataFrame:
    g = repay.sort_values(["account_id", "payment_month"]).groupby("account_id")
    out = g.agg(
        repay_num_months=("payment_month", "count"),
        repay_pct_missed=("is_missed", "mean"),
        repay_pct_late=("is_late", "mean"),
        repay_avg_dpd=("days_past_due", "mean"),
        repay_max_dpd=("days_past_due", "max"),
        repay_last_dpd=("days_past_due", "last"),
        repay_avg_underpayment=("underpayment", "mean"),
        repay_total_due=("emi_due", "sum"),
        repay_total_paid=("emi_paid", "sum"),
    ).reset_index()

    # Trend: average DPD in the most recent 3 months vs the first 3 months.
    def trend(sub: pd.DataFrame) -> float:
        sub = sub.sort_values("payment_month")
        first3 = sub["days_past_due"].head(3).mean()
        last3 = sub["days_past_due"].tail(3).mean()
        return last3 - first3

    trend_s = repay.sort_values(["account_id", "payment_month"]).groupby("account_id").apply(trend, include_groups=False)
    trend_s.name = "repay_dpd_trend"
    out = out.merge(trend_s.reset_index(), on="account_id", how="left")
    out["repay_payment_ratio"] = (out["repay_total_paid"] / out["repay_total_due"].replace(0, np.nan)).fillna(1.0)
    return out


def collections_features(coll: pd.DataFrame) -> pd.DataFrame:
    g = coll.groupby("account_id")
    out = g.agg(
        coll_num_events=("collection_event_id", "count"),
        coll_num_legal_notice=("action_type", lambda s: (s == "Legal Notice").sum()),
        coll_num_field_visit=("action_type", lambda s: (s == "Field Visit").sum()),
        coll_total_recovered=("amount_recovered", "sum"),
        coll_num_escalated=("outcome", lambda s: (s == "Escalated").sum()),
    ).reset_index()
    return out


def build_model_df(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Account-grain dataset for modeling + account-level dashboard view."""
    loans = tables["loan_accounts"]
    customers = tables["customers"]
    bureau_agg = bureau_features(tables["bureau_credit"])
    repay_agg = repayment_features(tables["repayment_history"])
    coll_agg = collections_features(tables["collections_events"])

    df = loans.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(bureau_agg, on="customer_id", how="left")
    df = df.merge(repay_agg, on="account_id", how="left")
    df = df.merge(coll_agg, on="account_id", how="left")

    # No bureau/repayment/collections history -> fill with 0 (genuine absence, not missingness to impute)
    fill_zero_cols = [c for c in df.columns if c.startswith(("bureau_", "repay_", "coll_"))]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    now = pd.Timestamp("2026-01-01")
    df["loan_age_months"] = ((now - df["origination_date"]).dt.days / 30.44).clip(lower=0)
    df["customer_tenure_years"] = ((now - df["customer_since"]).dt.days / 365.25).clip(lower=0)

    age_bins = [0, 25, 35, 45, 55, 65, 120]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    df["age_band"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

    inc_bins = [0, 25000, 50000, 75000, 100000, 150000, np.inf]
    inc_labels = ["<25k", "25-50k", "50-75k", "75-100k", "100-150k", "150k+"]
    df["income_band"] = pd.cut(df["annual_income"], bins=inc_bins, labels=inc_labels, right=True)
    return df


def build_customer_df(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Customer-grain dataset for portfolio profiling/segmentation."""
    customers = tables["customers"]
    loans = tables["loan_accounts"]
    bureau_agg = bureau_features(tables["bureau_credit"])

    loan_agg = loans.groupby("customer_id").agg(
        num_loans=("account_id", "count"),
        total_exposure=("loan_amount", "sum"),
        any_default=("target_default", "max"),
        default_rate_own=("target_default", "mean"),
    ).reset_index()

    df = customers.merge(loan_agg, on="customer_id", how="left")
    df = df.merge(bureau_agg, on="customer_id", how="left")
    for col in ["num_loans", "total_exposure", "any_default", "default_rate_own"]:
        df[col] = df[col].fillna(0)
    bureau_cols = [c for c in df.columns if c.startswith("bureau_")]
    df[bureau_cols] = df[bureau_cols].fillna(0)

    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    inc_bins = [0, 25000, 50000, 75000, 100000, 150000, np.inf]
    inc_labels = ["<25k", "25-50k", "50-75k", "75-100k", "100-150k", "150k+"]
    df["income_band"] = pd.cut(df["annual_income"], bins=inc_bins, labels=inc_labels, right=True)
    return df
