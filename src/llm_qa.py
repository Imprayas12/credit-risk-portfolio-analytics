"""
"Ask the Portfolio": grounded natural-language Q&A.

Design (see docs/architecture.pdf for the diagram):
- We NEVER send raw customer-level rows to the LLM. We pre-compute small
  aggregate tables (default rate by dimension, cross-tabs, model metrics,
  anomaly counts) in `build_grounding_context`, serialize that to compact
  JSON, and send *only that JSON* + the user's question to the LLM.
- The system prompt instructs the model to answer strictly from the
  supplied JSON, quote the exact figures, cite which table they came from,
  and reply with a fixed "I can't answer that from the available data"
  message if the question cannot be answered from the context.
- If no OPENAI_API_KEY is configured, we fall back to a deterministic
  rule-based matcher over the same grounding context (no free-text
  generation, but still correct, data-grounded answers for common
  questions), so the feature still "works" without a key.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from src.analytics import DIMENSIONS, default_rate_by, cross_default_rate, portfolio_summary

DECLINE_MSG = "I can't answer that from the available portfolio data."

SYSTEM_PROMPT = """You are "Ask the Portfolio", an assistant for a credit-risk analyst.
You will be given a JSON object called CONTEXT containing pre-computed, aggregated
statistics about a loan portfolio (default rates by dimension, cross-tabs, model
metrics, anomaly counts). You must answer the analyst's question using ONLY the
numbers present in CONTEXT.

Rules:
- Never invent or estimate a number that is not in CONTEXT.
- Always quote the exact figure(s) you used and name which part of CONTEXT it came from.
- Keep the answer short (1-3 sentences), plain English.
- If the question cannot be answered from CONTEXT, reply exactly: "%s"
""" % DECLINE_MSG


def build_grounding_context(model_df: pd.DataFrame, metrics: dict | None = None, anomaly_count: int | None = None) -> dict:
    ctx: dict = {"portfolio_summary": portfolio_summary(model_df)}
    for dim in DIMENSIONS:
        ctx[f"default_rate_by_{dim}"] = default_rate_by(model_df, dim).round(4).to_dict(orient="records")
    ctx["default_rate_by_employment_x_loan_type"] = (
        cross_default_rate(model_df, "employment_type", "loan_type").round(4).to_dict(orient="records")
    )
    if metrics:
        ctx["model_metrics"] = metrics
    if anomaly_count is not None:
        ctx["num_flagged_anomalies"] = anomaly_count
    return ctx


def _rule_based_answer(question: str, ctx: dict) -> str:
    q = question.lower()

    def fmt_rate(rec):
        return f"{rec['default_rate']:.1%} ({int(rec['accounts'])} accounts)"

    for dim in DIMENSIONS:
        key_tokens = dim.replace("_", " ").split()
        if any(tok in q for tok in key_tokens) or dim in q:
            recs = ctx.get(f"default_rate_by_{dim}")
            if not recs:
                continue
            recs_sorted = sorted(recs, key=lambda r: r["default_rate"], reverse=True)
            if "lowest" in q or "best" in q or "safest" in q:
                r = recs_sorted[-1]
            else:
                r = recs_sorted[0]
            return f"{r[dim]} has the highest default rate among {dim.replace('_',' ')} segments at {fmt_rate(r)}." \
                if "lowest" not in q else f"{r[dim]} has the lowest default rate among {dim.replace('_',' ')} segments at {fmt_rate(r)}."

    if "overall" in q and "default" in q:
        s = ctx["portfolio_summary"]
        return f"The overall portfolio default rate is {s['overall_default_rate']:.1%} across {s['total_accounts']} accounts."

    if "exposure" in q or "total loan" in q:
        s = ctx["portfolio_summary"]
        return f"Total exposure (sum of loan_amount) is ${s['total_exposure']:,.0f} across {s['total_accounts']} accounts."

    if "auc" in q or "gini" in q or " ks" in q or "ks stat" in q or "model perform" in q:
        if "model_metrics" in ctx:
            m = ctx["model_metrics"]
            parts = [f"{name}: AUC={v['auc']:.3f}, KS={v['ks']:.3f}, Gini={v['gini']:.3f}" for name, v in m.items()]
            return "Model performance — " + "; ".join(parts)

    if "anomal" in q or "flagged" in q:
        if "num_flagged_anomalies" in ctx:
            return f"{ctx['num_flagged_anomalies']} accounts are currently flagged by the anomaly detector for analyst review."

    if "how many customer" in q or "how many account" in q:
        s = ctx["portfolio_summary"]
        return f"The portfolio has {s['total_customers']} customers and {s['total_accounts']} loan accounts."

    return DECLINE_MSG


def answer_question(question: str, ctx: dict, api_key: str | None = None, model: str = "gpt-4o-mini") -> dict:
    """Returns {'answer': str, 'used_llm': bool}."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"answer": _rule_based_answer(question, ctx), "used_llm": False}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{json.dumps(ctx, default=str)}\n\nQUESTION: {question}"},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        return {"answer": answer, "used_llm": True}
    except Exception as e:  # noqa: BLE001
        fallback = _rule_based_answer(question, ctx)
        return {"answer": f"(LLM call failed, used fallback) {fallback}", "used_llm": False, "error": str(e)}
