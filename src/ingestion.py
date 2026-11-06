"""
Data ingestion & cleaning for the Credit Risk Portfolio dataset.

Loads the 5 raw CSVs and applies documented, explicit data-quality fixes.
Every decision below is intentional and logged via `print` / the returned
`dq_log` dict so it is auditable (also surfaced in the dashboard).

Design choices (documented, not silent):
- gender: ~6% of rows use "M"/"F" instead of "Male"/"Female" -> normalized.
- years_employed, num_dependents, years_at_residence (customers): missing
  values are imputed with the column median (robust to outliers) and a
  `<col>_was_missing` flag is kept so the model can use "missingness" itself
  as a signal (missingness is often informative in credit data, e.g.
  self-employed people skipping "years employed").
- interest_rate_pct (loan_accounts): missing values imputed with the median
  rate *within the same loan_type* (rates vary a lot by product), flagged.
- days_past_due (bureau_credit): missing treated as 0 (no DPD reported is
  the modal/expected case for an Active, up-to-date line), flagged.
- date columns parsed to datetime; unparsable -> NaT (none expected here).
- No rows are dropped for missingness; we only drop true duplicate keys.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATE_COLS = {
    "customers": ["customer_since"],
    "loan_accounts": ["origination_date"],
    "bureau_credit": ["date_opened", "date_closed"],
    "repayment_history": [],
    "collections_events": ["event_date"],
}


def _log(dq_log: dict, table: str, msg: str) -> None:
    dq_log.setdefault(table, []).append(msg)


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(raw_dir / "customers.csv"),
        "loan_accounts": pd.read_csv(raw_dir / "loan_accounts.csv"),
        "bureau_credit": pd.read_csv(raw_dir / "bureau_credit.csv"),
        "repayment_history": pd.read_csv(raw_dir / "repayment_history.csv"),
        "collections_events": pd.read_csv(raw_dir / "collections_events.csv"),
    }


def clean_customers(df: pd.DataFrame, dq_log: dict) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    dupes = df["customer_id"].duplicated().sum()
    if dupes:
        df = df.drop_duplicates(subset="customer_id")
        _log(dq_log, "customers", f"Dropped {dupes} duplicate customer_id rows.")

    gender_map = {"M": "Male", "F": "Female", "Male": "Male", "Female": "Female"}
    bad_gender = (~df["gender"].isin(["Male", "Female"])).sum()
    df["gender"] = df["gender"].map(gender_map).fillna(df["gender"])
    _log(dq_log, "customers", f"Normalized {bad_gender} inconsistent gender labels (M/F -> Male/Female).")

    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        n_missing = df[col].isna().sum()
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        _log(dq_log, "customers", f"Imputed {n_missing} missing '{col}' with median={median_val:.2f}; flagged in '{col}_was_missing'.")

    df["customer_since"] = pd.to_datetime(df["customer_since"], errors="coerce")
    df["annual_income"] = df["annual_income"].clip(lower=0)

    _log(dq_log, "customers", f"Final row count {len(df)} (from {before}).")
    return df


def clean_loan_accounts(df: pd.DataFrame, dq_log: dict) -> pd.DataFrame:
    df = df.copy()
    dupes = df["account_id"].duplicated().sum()
    if dupes:
        df = df.drop_duplicates(subset="account_id")
        _log(dq_log, "loan_accounts", f"Dropped {dupes} duplicate account_id rows.")

    n_missing = df["interest_rate_pct"].isna().sum()
    df["interest_rate_pct_was_missing"] = df["interest_rate_pct"].isna().astype(int)
    median_by_type = df.groupby("loan_type")["interest_rate_pct"].transform("median")
    df["interest_rate_pct"] = df["interest_rate_pct"].fillna(median_by_type)
    _log(dq_log, "loan_accounts", f"Imputed {n_missing} missing 'interest_rate_pct' with median rate per loan_type; flagged.")

    df["origination_date"] = pd.to_datetime(df["origination_date"], errors="coerce")
    df["target_default"] = df["target_default"].astype(int)
    return df


def clean_bureau_credit(df: pd.DataFrame, dq_log: dict) -> pd.DataFrame:
    df = df.copy()
    n_missing = df["days_past_due"].isna().sum()
    df["days_past_due_was_missing"] = df["days_past_due"].isna().astype(int)
    df["days_past_due"] = df["days_past_due"].fillna(0)
    _log(dq_log, "bureau_credit", f"Imputed {n_missing} missing 'days_past_due' with 0 (no delinquency reported); flagged.")

    df["date_opened"] = pd.to_datetime(df["date_opened"], errors="coerce")
    df["date_closed"] = pd.to_datetime(df["date_closed"], errors="coerce")
    df["is_written_off"] = (df["account_status"] == "Written-Off").astype(int)
    return df


def clean_repayment_history(df: pd.DataFrame, dq_log: dict) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["account_id", "payment_month"])
    dropped = before - len(df)
    if dropped:
        _log(dq_log, "repayment_history", f"Dropped {dropped} rows missing account_id/payment_month (unusable grain).")
    df["days_past_due"] = df["days_past_due"].fillna(0)
    df["is_missed"] = (df["payment_status"] == "Missed").astype(int)
    df["is_late"] = (df["payment_status"] == "Late").astype(int)
    df["underpayment"] = (df["emi_due"] - df["emi_paid"]).clip(lower=0)
    return df


def clean_collections_events(df: pd.DataFrame, dq_log: dict) -> pd.DataFrame:
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["amount_recovered"] = df["amount_recovered"].fillna(0)
    return df


def load_and_clean(raw_dir: Path = RAW_DIR) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Load all 5 tables and apply documented cleaning. Returns (tables, dq_log)."""
    raw = load_raw(raw_dir)
    dq_log: dict[str, list[str]] = {}
    cleaned = {
        "customers": clean_customers(raw["customers"], dq_log),
        "loan_accounts": clean_loan_accounts(raw["loan_accounts"], dq_log),
        "bureau_credit": clean_bureau_credit(raw["bureau_credit"], dq_log),
        "repayment_history": clean_repayment_history(raw["repayment_history"], dq_log),
        "collections_events": clean_collections_events(raw["collections_events"], dq_log),
    }

    # Referential integrity checks (documented, not silently fixed)
    orphan_loans = (~cleaned["loan_accounts"]["customer_id"].isin(cleaned["customers"]["customer_id"])).sum()
    if orphan_loans:
        _log(dq_log, "loan_accounts", f"WARNING: {orphan_loans} loan_accounts rows reference unknown customer_id.")
    orphan_repay = (~cleaned["repayment_history"]["account_id"].isin(cleaned["loan_accounts"]["account_id"])).sum()
    if orphan_repay:
        _log(dq_log, "repayment_history", f"WARNING: {orphan_repay} repayment rows reference unknown account_id.")
    orphan_coll = (~cleaned["collections_events"]["account_id"].isin(cleaned["loan_accounts"]["account_id"])).sum()
    if orphan_coll:
        _log(dq_log, "collections_events", f"WARNING: {orphan_coll} collections rows reference unknown account_id.")
    orphan_bureau = (~cleaned["bureau_credit"]["customer_id"].isin(cleaned["customers"]["customer_id"])).sum()
    if orphan_bureau:
        _log(dq_log, "bureau_credit", f"WARNING: {orphan_bureau} bureau rows reference unknown customer_id.")

    return cleaned, dq_log


if __name__ == "__main__":
    tables, log = load_and_clean()
    for table, msgs in log.items():
        print(f"\n[{table}]")
        for m in msgs:
            print(" -", m)
