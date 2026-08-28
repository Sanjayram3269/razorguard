from __future__ import annotations

import numpy as np
import pandas as pd

from razorguard.config import (
    ACCOUNTS_PATH,
    CHARGEBACKS_PATH,
    GENERATED_DATA,
    RANDOM_SEED,
    TRANSACTIONS_PATH,
)

COUNTRIES = np.array(["IN", "US", "GB", "SG", "AE", "AU"])
PAYMENT_METHODS = np.array(["card", "upi", "wallet", "netbanking"])
MERCHANT_CATEGORIES = np.array(["fashion", "electronics", "food", "travel", "digital"])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_accounts(rng: np.random.Generator, n_accounts: int) -> pd.DataFrame:
    created = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 420, n_accounts), unit="D"
    )
    accounts = pd.DataFrame(
        {
            "account_id": [f"A{i:06d}" for i in range(n_accounts)],
            "created_at": created,
            "home_country": rng.choice(COUNTRIES, n_accounts, p=[0.55, .15, .10, .08, .07, .05]),
            "device_id": [f"D{i:06d}" for i in rng.integers(0, int(n_accounts * 0.82), n_accounts)],
            "account_segment": rng.choice(
                ["new", "standard", "premium"],
                n_accounts,
                p=[0.18, 0.67, 0.15],
            ),
        }
    )
    return accounts


def generate_transactions(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    n_transactions: int,
) -> pd.DataFrame:
    account_idx = rng.integers(0, len(accounts), n_transactions)
    accounts_used = accounts.iloc[account_idx].reset_index(drop=True)

    start = pd.Timestamp("2026-01-01")
    timestamps = start + pd.to_timedelta(
        np.sort(rng.integers(0, 180 * 24 * 60 * 60, n_transactions)),
        unit="s",
    )

    base_amount = rng.lognormal(mean=np.log(140), sigma=0.85, size=n_transactions)
    segment_multiplier = accounts_used["account_segment"].map(
        {"new": 0.75, "standard": 1.0, "premium": 2.8}
    ).to_numpy()

    amount = np.clip(base_amount * segment_multiplier, 20, 25000)

    ip_country = rng.choice(COUNTRIES, n_transactions, p=[0.62, .12, .09, .07, .06, .04])
    shipping_country = accounts_used["home_country"].to_numpy().copy()

    # Some legitimate cross-border shopping exists.
    cross_border = rng.random(n_transactions) < 0.08
    shipping_country[cross_border] = rng.choice(COUNTRIES, cross_border.sum())

    payment_method = rng.choice(
        PAYMENT_METHODS, n_transactions, p=[0.48, .32, .12, .08]
    )
    category = rng.choice(MERCHANT_CATEGORIES, n_transactions, p=[.27, .20, .24, .12, .17])

    hour = pd.Series(timestamps).dt.hour.to_numpy()
    night = ((hour <= 5) | (hour >= 23)).astype(int)

    account_age_days = (
        pd.Series(timestamps).dt.normalize().to_numpy()
        - accounts_used["created_at"].dt.normalize().to_numpy()
    ) / np.timedelta64(1, "D")
    account_age_days = np.maximum(account_age_days.astype(float), 0)

    location_mismatch = (
        ip_country != shipping_country
    ).astype(int)

    # Latent risk process. This creates ground truth from multiple interacting
    # mechanisms rather than one obvious rule.
    amount_log = np.log1p(amount)
    high_amount = (amount > 650).astype(int)
    very_new = (account_age_days < 14).astype(int)
    shared_device_signal = pd.Series(accounts_used["device_id"]).duplicated(keep=False).astype(int).to_numpy()

    risk_logit = (
        -5.0
        + 0.55 * (amount_log - np.log(150))
        + 1.05 * high_amount
        + 1.25 * very_new
        + 1.00 * location_mismatch
        + 0.75 * night
        + 0.55 * (payment_method == "card").astype(int)
        + 0.65 * shared_device_signal
        + 0.35 * (category == "electronics").astype(int)
    )

    # A small fraction follows a high-velocity burst pattern.
    burst_group = rng.random(n_transactions) < 0.035
    risk_logit += 1.15 * burst_group.astype(int)

    chargeback_probability = sigmoid(risk_logit)
    is_chargeback = rng.random(n_transactions) < chargeback_probability

    tx = pd.DataFrame(
        {
            "transaction_id": [f"T{i:08d}" for i in range(n_transactions)],
            "account_id": accounts_used["account_id"].to_numpy(),
            "merchant_id": [f"M{i:04d}" for i in rng.integers(0, 300, n_transactions)],
            "timestamp": timestamps,
            "amount": np.round(amount, 2),
            "currency": "INR",
            "payment_method": payment_method,
            "merchant_category": category,
            "ip_country": ip_country,
            "shipping_country": shipping_country,
            "device_id": accounts_used["device_id"].to_numpy(),
            "is_chargeback": is_chargeback.astype(int),
        }
    )

    return tx


def generate_chargebacks(
    rng: np.random.Generator, transactions: pd.DataFrame
) -> pd.DataFrame:
    cb = transactions.loc[transactions["is_chargeback"] == 1, ["transaction_id", "timestamp"]].copy()
    cb["chargeback_at"] = cb["timestamp"] + pd.to_timedelta(
        rng.integers(3, 35, len(cb)), unit="D"
    )
    cb["reason_code"] = rng.choice(
        ["fraud", "card_not_present", "duplicate", "service_not_received"],
        len(cb),
        p=[.52, .24, .08, .16],
    )
    return cb[["transaction_id", "chargeback_at", "reason_code"]]


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    accounts = generate_accounts(rng, n_accounts=5000)
    transactions = generate_transactions(rng, accounts, n_transactions=50000)
    chargebacks = generate_chargebacks(rng, transactions)

    accounts.to_parquet(ACCOUNTS_PATH, index=False)
    transactions.to_parquet(TRANSACTIONS_PATH, index=False)
    chargebacks.to_parquet(CHARGEBACKS_PATH, index=False)

    print(f"accounts={len(accounts):,}")
    print(f"transactions={len(transactions):,}")
    print(f"chargebacks={len(chargebacks):,}")
    print(f"chargeback_rate={transactions.is_chargeback.mean():.4%}")
    print(f"written_to={GENERATED_DATA}")


if __name__ == "__main__":
    main()
