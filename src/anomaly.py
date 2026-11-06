"""
Anomaly detection: flags accounts whose repayment/bureau behaviour is
unusual versus their own loan_type + employment_type peer segment, using
an Isolation Forest on standardized behavioural features, plus a simple
own-history check (sudden DPD deterioration).

For every flagged account we return the feature(s) that drove the flag
(largest |z-score| vs peer segment mean) so an analyst sees *why*.

Trigger: isolation-forest anomaly score in the worst 5% of its peer segment,
OR a jump in days-past-due of 30+ days between the first and last 3 months
of repayment history.
False positive: a genuinely low-risk account that happens to have a thin/
atypical history (e.g. very new loan, small peer segment) gets flagged.
False negative: a slowly-deteriorating account whose behaviour stays close
to its peer segment average (e.g. an already-risky segment) is missed.
Human-in-the-loop: analyst reviews each flag in the dashboard and records
Confirm (genuine risk, escalate) or Dismiss (false positive, no action).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
    "repay_avg_dpd", "repay_pct_missed", "repay_pct_late", "repay_dpd_trend",
    "repay_avg_underpayment", "bureau_avg_dpd", "bureau_utilization", "coll_num_events",
]
SEGMENT_COLS = ["loan_type", "employment_type"]
DPD_JUMP_THRESHOLD = 30
CONTAMINATION = 0.05


def _segment_z_scores(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    z = df[features].copy()
    for seg, grp in df.groupby(SEGMENT_COLS, observed=True).groups.items():
        idx = grp
        sub = df.loc[idx, features]
        mu, sigma = sub.mean(), sub.std(ddof=0).replace(0, 1)
        z.loc[idx, features] = (sub - mu) / sigma
    return z.fillna(0)


def detect_anomalies(model_df: pd.DataFrame, contamination: float = CONTAMINATION) -> pd.DataFrame:
    df = model_df.copy()
    z = _segment_z_scores(df, ANOMALY_FEATURES)

    scores = pd.Series(0.0, index=df.index)
    for seg, grp in df.groupby(SEGMENT_COLS, observed=True).groups.items():
        idx = grp
        if len(idx) < 20:
            # Too small a peer group for a stable model; mark as not-fitted (score 0 -> not anomalous via IF)
            continue
        X = z.loc[idx, ANOMALY_FEATURES].values
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        iso.fit(X)
        # decision_function: higher = more normal. Flip sign so higher = more anomalous.
        scores.loc[idx] = -iso.decision_function(X)

    df["anomaly_score"] = scores
    threshold = df.groupby(SEGMENT_COLS, observed=True)["anomaly_score"].transform(
        lambda s: s.quantile(1 - contamination) if len(s) >= 20 else np.inf
    )
    df["flag_peer_outlier"] = (df["anomaly_score"] >= threshold) & (df["anomaly_score"] > 0)
    df["flag_dpd_jump"] = df["repay_dpd_trend"] >= DPD_JUMP_THRESHOLD
    df["is_anomaly"] = df["flag_peer_outlier"] | df["flag_dpd_jump"]

    def reason(row: pd.Series) -> str:
        reasons = []
        if row["flag_dpd_jump"]:
            reasons.append(f"DPD jumped {row['repay_dpd_trend']:.0f} days vs early loan life")
        if row["flag_peer_outlier"]:
            rz = z.loc[row.name, ANOMALY_FEATURES]
            top = rz.abs().sort_values(ascending=False).head(2)
            for feat in top.index:
                direction = "higher" if rz[feat] > 0 else "lower"
                reasons.append(f"{feat} is {direction} than peers (z={rz[feat]:.1f})")
        return "; ".join(reasons) if reasons else "Not flagged"

    df["anomaly_reason"] = df.apply(reason, axis=1)
    return df


def flagged_accounts(model_df: pd.DataFrame, contamination: float = CONTAMINATION) -> pd.DataFrame:
    scored = detect_anomalies(model_df, contamination)
    cols = [
        "account_id", "customer_id", "loan_type", "employment_type", "target_default",
        "anomaly_score", "flag_peer_outlier", "flag_dpd_jump", "anomaly_reason",
        "repay_avg_dpd", "repay_pct_missed", "repay_dpd_trend", "bureau_utilization", "coll_num_events",
    ]
    return scored.loc[scored["is_anomaly"], cols].sort_values("anomaly_score", ascending=False)
