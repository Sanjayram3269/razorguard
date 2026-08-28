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
from razorguard.data.scenarios import ScenarioConfig, assign_account_scenarios, attach_behavior_state
from razorguard.data.validation import validate_account_scenarios, validate_transactions

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
            "home_country": rng.choice(
                COUNTRIES, n_accounts, p=[0.55, 0.15, 0.10, 0.08, 0.07, 0.05]
            ),
            "device_id": [
                f"D{i:06d}"
                for i in rng.integers(0, int(n_accounts * 0.82), n_accounts)
            ],
            "account_segment": rng.choice(
                ["new", "standard", "premium"],
                n_accounts,
                p=[0.18, 0.67, 0.15],
            ),
        }
    )
    accounts = assign_account_scenarios(
        rng, accounts, ScenarioConfig()
    )
    accounts = attach_behavior_state(rng, accounts)

    # Explicit infrastructure relationships make graph-like behavior emerge
    # from the world instead of being inferred from labels.
    coordinated = accounts["scenario"].eq("coordinated")
    coordinated_ids = rng.choice(
        [f"CD{i:04d}" for i in range(max(1, coordinated.sum() // 3))],
        size=coordinated.sum(),
    )
    accounts.loc[coordinated, "device_id"] = coordinated_ids

    shared = accounts["scenario"].eq("shared_infrastructure")
    shared_ids = rng.choice(
        [f"SH{i:04d}" for i in range(max(1, shared.sum() // 4))],
        size=shared.sum(),
    )
    accounts.loc[shared, "device_id"] = shared_ids

    # Compromised accounts can switch to a previously unseen device after the
    # hidden compromise point. The device itself remains an observable signal.
    accounts["secondary_device_id"] = [
        f"X{i:06d}" for i in range(n_accounts)
    ]
    return accounts


def _transaction_times(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    account_idx: np.ndarray,
    n_transactions: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Create a chronological event stream, including account-local bursts."""
    start = pd.Timestamp("2026-01-01")
    horizon_seconds = 180 * 24 * 60 * 60
    timestamps = start + pd.to_timedelta(
        rng.integers(0, horizon_seconds, n_transactions), unit="s"
    )

    # A subset of burst accounts receive clustered events. We preserve the
    # global time range and sort only after all event generation is complete.
    scenarios = accounts.iloc[account_idx]["scenario"].to_numpy()
    burst_mask = scenarios == "burst"
    burst_indices = np.flatnonzero(burst_mask)
    if len(burst_indices):
        burst_centers = rng.integers(0, horizon_seconds, len(burst_indices))
        offsets = rng.integers(-15 * 60, 15 * 60 + 1, len(burst_indices))
        raw_seconds = np.asarray(
            timestamps.astype("int64") // 1_000_000_000
        ) - int(start.timestamp())
        raw_seconds[burst_indices] = np.clip(
            burst_centers + offsets, 0, horizon_seconds - 1
        )
        timestamps = start + pd.to_timedelta(raw_seconds, unit="s")

    order = np.argsort(np.asarray(timestamps, dtype="datetime64[ns]"))
    return pd.DatetimeIndex(timestamps[order]), order


def generate_transactions(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    n_transactions: int,
) -> pd.DataFrame:
    account_idx = rng.integers(0, len(accounts), n_transactions)
    timestamps, order = _transaction_times(rng, accounts, account_idx, n_transactions)
    account_idx = account_idx[order]
    accounts_used = accounts.iloc[account_idx].reset_index(drop=True)

    scenario = accounts_used["scenario"].to_numpy()
    baseline = accounts_used["behavior_baseline"].to_numpy(dtype=float)

    # Base spend is behavior-driven. High-value legitimate customers spend
    # more without receiving a fraudulent label merely because of amount.
    amount = baseline * rng.lognormal(mean=0.0, sigma=0.55, size=n_transactions)
    amount *= np.where(scenario == "new_account", 0.75, 1.0)
    amount *= np.where(scenario == "high_value_legitimate", 1.15, 1.0)

    # Compromise creates a regime shift rather than a static risky identity.
    elapsed_fraction = (
        (timestamps - timestamps.min()).total_seconds()
        / max((timestamps.max() - timestamps.min()).total_seconds(), 1.0)
    )
    compromised = scenario == "compromised"
    post_compromise = compromised & (
        elapsed_fraction >= accounts_used["compromise_start_fraction"].to_numpy()
    )
    amount *= np.where(
        post_compromise,
        accounts_used["compromise_multiplier"].to_numpy(dtype=float),
        1.0,
    )

    # Burst accounts increase transaction density and spend somewhat, but
    # burst behavior is still probabilistic rather than a deterministic label.
    burst = scenario == "burst"
    amount *= np.where(burst, rng.uniform(1.15, 2.0, n_transactions), 1.0)
    amount = np.clip(amount, 20, 25_000)

    ip_country = rng.choice(
        COUNTRIES, n_transactions, p=[0.62, 0.12, 0.09, 0.07, 0.06, 0.04]
    )
    shipping_country = accounts_used["home_country"].to_numpy().copy()
    cross_border = rng.random(n_transactions) < 0.08
    shipping_country[cross_border] = rng.choice(COUNTRIES, cross_border.sum())

    # Compromised accounts become more likely to transact from a country that
    # differs from their shipping/home context after the regime shift.
    compromise_location = post_compromise & (rng.random(n_transactions) < 0.62)
    if compromise_location.any():
        foreign = rng.choice(COUNTRIES, compromise_location.sum())
        home = shipping_country[compromise_location]
        foreign = np.where(foreign == home, "US", foreign)
        ip_country[compromise_location] = foreign

    payment_method = rng.choice(
        PAYMENT_METHODS, n_transactions, p=[0.48, 0.32, 0.12, 0.08]
    )
    category = rng.choice(
        MERCHANT_CATEGORIES, n_transactions, p=[0.27, 0.20, 0.24, 0.12, 0.17]
    )

    hour = pd.Series(timestamps).dt.hour.to_numpy()
    night = ((hour <= 5) | (hour >= 23)).astype(int)

    account_age_days = (
        pd.Series(timestamps).dt.normalize().to_numpy()
        - accounts_used["created_at"].dt.normalize().to_numpy()
    ) / np.timedelta64(1, "D")
    account_age_days = np.maximum(account_age_days.astype(float), 0)

    location_mismatch = (ip_country != shipping_country).astype(int)
    amount_log = np.log1p(amount)
    high_amount = (amount > 650).astype(int)
    very_new = (account_age_days < 14).astype(int)

    # Observable infrastructure signals. Shared infrastructure is intentionally
    # assigned a smaller risk contribution than coordinated infrastructure.
    shared_device = pd.Series(accounts_used["device_id"]).duplicated(
        keep=False
    ).to_numpy()
    coordinated_signal = (scenario == "coordinated").astype(int)
    legitimate_shared_signal = (scenario == "shared_infrastructure").astype(int)

    risk_logit = (
        -5.35
        + 0.38 * (amount_log - np.log(150))
        + 0.80 * high_amount
        + 0.75 * very_new
        + 0.72 * location_mismatch
        + 0.50 * night
        + 0.35 * (payment_method == "card").astype(int)
        + 0.32 * shared_device.astype(int)
        + 0.95 * coordinated_signal
        + 1.55 * post_compromise.astype(int)
        + 0.85 * burst.astype(int)
        - 0.55 * legitimate_shared_signal
    )

    # Add small irreducible noise so identical behavioral profiles don't always
    # receive identical outcomes.
    risk_logit += rng.normal(0, 0.28, n_transactions)
    chargeback_probability = sigmoid(risk_logit)
    is_chargeback = rng.random(n_transactions) < chargeback_probability

    device_id = accounts_used["device_id"].to_numpy().copy()
    device_id[post_compromise] = accounts_used.loc[
        post_compromise, "secondary_device_id"
    ].to_numpy()

    merchant_pool = rng.integers(0, 300, n_transactions)

    tx = pd.DataFrame(
        {
            "transaction_id": [f"T{i:08d}" for i in range(n_transactions)],
            "account_id": accounts_used["account_id"].to_numpy(),
            "merchant_id": [f"M{i:04d}" for i in merchant_pool],
            "timestamp": timestamps,
            "amount": np.round(amount, 2),
            "currency": "INR",
            "payment_method": payment_method,
            "merchant_category": category,
            "ip_country": ip_country,
            "shipping_country": shipping_country,
            "device_id": device_id,
            "is_chargeback": is_chargeback.astype(int),
        }
    )
    return tx.sort_values("timestamp").reset_index(drop=True)


def generate_chargebacks(
    rng: np.random.Generator, transactions: pd.DataFrame
) -> pd.DataFrame:
    cb = transactions.loc[
        transactions["is_chargeback"] == 1, ["transaction_id", "timestamp"]
    ].copy()
    cb["chargeback_at"] = cb["timestamp"] + pd.to_timedelta(
        rng.integers(3, 35, len(cb)), unit="D"
    )
    cb["reason_code"] = rng.choice(
        ["fraud", "card_not_present", "duplicate", "service_not_received"],
        len(cb),
        p=[0.52, 0.24, 0.08, 0.16],
    )
    return cb[["transaction_id", "chargeback_at", "reason_code"]]


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    accounts = generate_accounts(rng, n_accounts=5000)
    transactions = generate_transactions(rng, accounts, n_transactions=50000)
    chargebacks = generate_chargebacks(rng, transactions)

    validate_account_scenarios(accounts)
    validate_transactions(transactions)

    accounts.to_parquet(ACCOUNTS_PATH, index=False)
    transactions.to_parquet(TRANSACTIONS_PATH, index=False)
    chargebacks.to_parquet(CHARGEBACKS_PATH, index=False)

    scenario_counts = accounts["scenario"].value_counts().sort_index()
    print(f"accounts={len(accounts):,}")
    print(f"transactions={len(transactions):,}")
    print(f"chargebacks={len(chargebacks):,}")
    print(f"chargeback_rate={transactions.is_chargeback.mean():.4%}")
    print("scenario_counts=")
    for scenario_name, count in scenario_counts.items():
        print(f"  {scenario_name}={count:,}")
    print(f"written_to={GENERATED_DATA}")


if __name__ == "__main__":
    main()
