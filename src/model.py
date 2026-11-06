"""
Default model: a Logistic Regression baseline and an XGBoost challenger,
trained on engineered customer/bureau/repayment/collections features.

Reports AUC, KS, Gini and feature importance (coefficients for LR,
gain-based importance + SHAP for XGBoost). Artifacts are saved to
`artifacts/` for the dashboard to load without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

NUMERIC_FEATURES = [
    "age", "years_employed", "annual_income", "num_dependents", "years_at_residence",
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag",
    "bureau_num_lines", "bureau_total_sanctioned", "bureau_total_outstanding",
    "bureau_avg_dpd", "bureau_max_dpd", "bureau_num_written_off", "bureau_num_active",
    "bureau_utilization", "repay_num_months", "repay_pct_missed", "repay_pct_late",
    "repay_avg_dpd", "repay_max_dpd", "repay_last_dpd", "repay_avg_underpayment",
    "repay_dpd_trend", "repay_payment_ratio", "coll_num_events", "coll_num_legal_notice",
    "coll_num_field_visit", "coll_total_recovered", "coll_num_escalated",
    "loan_age_months", "customer_tenure_years",
    "years_employed_was_missing", "num_dependents_was_missing",
    "years_at_residence_was_missing", "interest_rate_pct_was_missing",
]
CATEGORICAL_FEATURES = [
    "gender", "marital_status", "education_level", "employment_type",
    "occupation_sector", "region", "city_tier", "housing_type",
    "loan_type", "loan_purpose", "disbursal_channel",
]
TARGET = "target_default"


def _ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(model_df: pd.DataFrame, random_state: int = 42) -> dict:
    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_df[TARGET]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, model_df.index, test_size=0.25, stratify=y, random_state=random_state
    )

    results = {}
    fitted = {}

    # --- Logistic Regression baseline ---
    lr_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_scores = lr_pipe.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_scores)
    results["logistic_regression"] = {
        "auc": lr_auc, "ks": _ks_stat(y_test, lr_scores), "gini": 2 * lr_auc - 1,
    }
    fitted["logistic_regression"] = lr_pipe

    # --- XGBoost challenger ---
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="auc", random_state=random_state,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    xgb_scores = xgb_pipe.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_scores)
    results["xgboost"] = {
        "auc": xgb_auc, "ks": _ks_stat(y_test, xgb_scores), "gini": 2 * xgb_auc - 1,
    }
    fitted["xgboost"] = xgb_pipe

    # Feature importance (XGBoost gain, mapped back to encoded names)
    cat_names = list(xgb_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    feat_names = NUMERIC_FEATURES + cat_names
    importances = xgb_pipe.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    # Test-set predictions for the account-level dashboard view
    test_predictions = model_df.loc[idx_test, ["account_id", "customer_id", TARGET]].copy()
    test_predictions["risk_score"] = xgb_scores

    return {
        "metrics": results,
        "feature_importance": importance_df,
        "test_predictions": test_predictions,
        "models": fitted,
    }


def score_all_with_drivers(result: dict, model_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Score every account with the XGBoost model and attach the top-N SHAP
    drivers (feature + direction) so the dashboard's account view can show
    'why' a score is high or low, without recomputing per click."""
    import shap

    xgb_pipe = result["models"]["xgboost"]
    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_trans = xgb_pipe.named_steps["prep"].transform(X)
    cat_names = list(xgb_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    feat_names = NUMERIC_FEATURES + cat_names

    scores = xgb_pipe.named_steps["clf"].predict_proba(X_trans)[:, 1]
    explainer = shap.TreeExplainer(xgb_pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_trans)

    drivers = []
    for row in shap_values:
        order = np.argsort(-np.abs(row))[:top_n]
        parts = []
        for i in order:
            direction = "increases" if row[i] > 0 else "decreases"
            parts.append(f"{feat_names[i]} {direction} risk")
        drivers.append("; ".join(parts))

    out = model_df[["account_id", "customer_id", "target_default"]].copy()
    out["risk_score"] = scores
    out["top_drivers"] = drivers
    return out


def save_artifacts(result: dict, out_dir: Path = ARTIFACT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["models"]["xgboost"], out_dir / "xgb_model.joblib")
    joblib.dump(result["models"]["logistic_regression"], out_dir / "lr_model.joblib")
    result["feature_importance"].to_csv(out_dir / "feature_importance.csv", index=False)
    result["test_predictions"].to_csv(out_dir / "test_predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result["metrics"], f, indent=2)


if __name__ == "__main__":
    from src.ingestion import load_and_clean
    from src.features import build_model_df

    tables, _ = load_and_clean()
    model_df = build_model_df(tables)
    result = train_and_evaluate(model_df)
    print(json.dumps(result["metrics"], indent=2))
    print(result["feature_importance"].head(15))
    save_artifacts(result)
