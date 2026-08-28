from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id",
    "account_id",
    "merchant_id",
    "device_id",
    "timestamp",
    "amount",
    "is_chargeback",
}


SCENARIOS = {
    "normal",
    "new_account",
    "high_value_legitimate",
    "compromised",
    "burst",
    "coordinated",
    "shared_infrastructure",
}


def validate_transactions(transactions: pd.DataFrame) -> None:
    """Fail fast on structural or temporal problems in an E2 dataset."""
    missing = REQUIRED_COLUMNS - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if transactions["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique")

    if not transactions["timestamp"].is_monotonic_increasing:
        raise ValueError("transactions must be sorted chronologically")

    if (transactions["amount"] <= 0).any():
        raise ValueError("transaction amounts must be positive")

    labels = set(transactions["is_chargeback"].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError("is_chargeback must contain only 0/1 labels")


def validate_account_scenarios(accounts: pd.DataFrame) -> None:
    required = {"account_id", "scenario", "behavior_baseline"}
    missing = required - set(accounts.columns)
    if missing:
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")

    unknown = set(accounts["scenario"].unique()) - SCENARIOS
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

    if accounts["account_id"].duplicated().any():
        raise ValueError("account_id must be unique")

    if (accounts["behavior_baseline"] <= 0).any():
        raise ValueError("behavior_baseline must be positive")


def validate_future_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Ensure chronological evaluation has no temporal overlap."""
    if train["timestamp"].max() >= validation["timestamp"].min():
        raise ValueError("training and validation periods overlap")
    if validation["timestamp"].max() >= test["timestamp"].min():
        raise ValueError("validation and test periods overlap")
