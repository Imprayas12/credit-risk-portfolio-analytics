"""
One-command build pipeline: raw CSVs -> cleaned tables -> engineered
features -> trained model -> anomaly flags -> cached artifacts that the
Streamlit dashboard reads at startup (so the app doesn't retrain on every
reload).

Run: python -m src.build
"""
from __future__ import annotations

import json
from pathlib import Path

from src.ingestion import load_and_clean
from src.features import build_model_df, build_customer_df
from src.model import train_and_evaluate, save_artifacts, score_all_with_drivers, ARTIFACT_DIR
from src.anomaly import detect_anomalies

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    print("1/5 Loading & cleaning raw tables...")
    tables, dq_log = load_and_clean()

    print("2/5 Engineering features...")
    model_df = build_model_df(tables)
    customer_df = build_customer_df(tables)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(PROCESSED_DIR / "model_df.parquet", index=False)
    customer_df.to_parquet(PROCESSED_DIR / "customer_df.parquet", index=False)
    with open(PROCESSED_DIR / "dq_log.json", "w") as f:
        json.dump(dq_log, f, indent=2)

    print("3/5 Training default model (Logistic Regression + XGBoost)...")
    result = train_and_evaluate(model_df)
    save_artifacts(result, ARTIFACT_DIR)
    print(json.dumps(result["metrics"], indent=2))

    print("3b/5 Scoring all accounts with SHAP drivers...")
    all_scores = score_all_with_drivers(result, model_df)
    all_scores.to_parquet(ARTIFACT_DIR / "account_scores.parquet", index=False)

    print("4/5 Running anomaly detection...")
    scored = detect_anomalies(model_df)
    scored.to_parquet(ARTIFACT_DIR / "scored_accounts.parquet", index=False)
    n_flagged = int(scored["is_anomaly"].sum())
    print(f"   {n_flagged} accounts flagged for review.")

    print("5/5 Done. Artifacts written to data/processed/ and artifacts/.")


if __name__ == "__main__":
    main()
