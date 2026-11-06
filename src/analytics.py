"""
Portfolio analytics: profiling cuts across income, employment type, region,
age and loan type, plus a function that turns the numbers into plain-English
insights with a "so what" — these same functions back the dashboard's
portfolio view and the "Ask the Portfolio" grounding layer.
"""
from __future__ import annotations

import pandas as pd

DIMENSIONS = ["loan_type", "employment_type", "region", "income_band", "age_band"]


def default_rate_by(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    out = df.groupby(dim, observed=True).agg(
        accounts=("account_id", "count") if "account_id" in df.columns else ("customer_id", "count"),
        default_rate=("target_default", "mean") if "target_default" in df.columns else ("any_default", "mean"),
        exposure=("loan_amount", "sum") if "loan_amount" in df.columns else ("total_exposure", "sum"),
    ).reset_index().sort_values("default_rate", ascending=False)
    return out


def cross_default_rate(df: pd.DataFrame, dim1: str, dim2: str) -> pd.DataFrame:
    target = "target_default" if "target_default" in df.columns else "any_default"
    out = df.groupby([dim1, dim2], observed=True).agg(
        accounts=(target, "count"),
        default_rate=(target, "mean"),
    ).reset_index()
    return out


def portfolio_summary(model_df: pd.DataFrame) -> dict:
    return {
        "total_accounts": int(len(model_df)),
        "total_customers": int(model_df["customer_id"].nunique()),
        "total_exposure": float(model_df["loan_amount"].sum()),
        "overall_default_rate": float(model_df["target_default"].mean()),
    }


def generate_insights(model_df: pd.DataFrame, customer_df: pd.DataFrame) -> list[dict]:
    """Return 4-5 insights with a concrete 'so what', at least one cross-dim."""
    insights = []

    by_emp = default_rate_by(model_df, "employment_type")
    hi, lo = by_emp.iloc[0], by_emp.iloc[-1]
    insights.append({
        "title": "Employment type is a strong default driver",
        "detail": f"{hi['employment_type']} loans default at {hi['default_rate']:.1%} vs "
                  f"{lo['employment_type']} at {lo['default_rate']:.1%} ({hi['accounts']:.0f} vs {lo['accounts']:.0f} accounts).",
        "so_what": "Tighten underwriting (lower LTV / extra income proof) or reprice for the highest-risk "
                   "employment segment rather than applying a flat rate across all employment types.",
    })

    by_loan = default_rate_by(model_df, "loan_type")
    hi = by_loan.iloc[0]
    insights.append({
        "title": f"{hi['loan_type']} is the riskiest product",
        "detail": f"{hi['loan_type']} defaults at {hi['default_rate']:.1%}, carrying ${hi['exposure']:,.0f} of exposure "
                  f"across {hi['accounts']:.0f} accounts.",
        "so_what": "Review pricing/collateral requirements for this product line; it is disproportionately "
                   "contributing to expected loss relative to its share of the book.",
    })

    by_income = default_rate_by(model_df, "income_band")
    low_income = by_income.sort_values("income_band").iloc[0] if "income_band" in model_df else None
    worst = by_income.iloc[0]
    insights.append({
        "title": "Lower income bands default more often",
        "detail": f"Default rate is highest in the {worst['income_band']} income band at {worst['default_rate']:.1%}, "
                  f"vs the portfolio average of {model_df['target_default'].mean():.1%}.",
        "so_what": "Income-tiered affordability checks (e.g. debt-to-income caps) could reduce losses in the "
                   "lowest income bands without broadly restricting credit access.",
    })

    cross = cross_default_rate(model_df, "employment_type", "loan_type")
    cross = cross[cross["accounts"] >= 30].sort_values("default_rate", ascending=False)
    if len(cross):
        top = cross.iloc[0]
        insights.append({
            "title": "Cross-dimensional hotspot: employment type x loan type",
            "detail": f"{top['employment_type']} customers taking a {top['loan_type']} default at "
                      f"{top['default_rate']:.1%} ({top['accounts']:.0f} accounts) — the highest combination "
                      f"in the portfolio (min. 30 accounts).",
            "so_what": "This single combination should get its own underwriting rule or pricing add-on "
                       "rather than being averaged into the broader employment-type or loan-type policy.",
        })

    by_region = default_rate_by(model_df, "region")
    hi, lo = by_region.iloc[0], by_region.iloc[-1]
    insights.append({
        "title": "Regional spread in default rates",
        "detail": f"{hi['region']} region has the highest default rate at {hi['default_rate']:.1%}, "
                  f"{(hi['default_rate'] - lo['default_rate']) * 100:.1f}pp above {lo['region']} "
                  f"({lo['default_rate']:.1%}).",
        "so_what": "Regional risk teams in the higher-default region should be resourced for more proactive "
                   "collections and tighter new-account monitoring.",
    })

    return insights
