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
from razorguard.data.event_process import generate_account_event_stream
from razorguard.data.scenarios import (
    ScenarioConfig,
    assign_account_scenarios,
    attach_behavior_state,
)
from razorguard.data.validation import (
    validate_account_scenarios,
    validate_transactions,
)


COUNTRIES = np.array(
    ["IN", "US", "GB", "SG", "AE", "AU"]
)

PAYMENT_METHODS = np.array(
    ["card", "upi", "wallet", "netbanking"]
)

MERCHANT_CATEGORIES = np.array(
    ["fashion", "electronics", "food", "travel", "digital"]
)

N_MERCHANTS = 300


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transformation."""
    return 1.0 / (
        1.0 + np.exp(-np.clip(values, -30.0, 30.0))
    )


def generate_accounts(
    rng: np.random.Generator,
    n_accounts: int,
) -> pd.DataFrame:
    """Generate account identities and hidden behavioral state."""

    created_at = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            rng.integers(
                0,
                420,
                n_accounts,
            ),
            unit="D",
        )
    )

    accounts = pd.DataFrame(
        {
            "account_id": [
                f"A{i:06d}"
                for i in range(n_accounts)
            ],
            "created_at": created_at,
            "home_country": rng.choice(
                COUNTRIES,
                size=n_accounts,
                p=[
                    0.55,
                    0.15,
                    0.10,
                    0.08,
                    0.07,
                    0.05,
                ],
            ),
            "device_id": [
                f"D{i:06d}"
                for i in rng.integers(
                    0,
                    int(n_accounts * 0.82),
                    n_accounts,
                )
            ],
            "account_segment": rng.choice(
                ["new", "standard", "premium"],
                size=n_accounts,
                p=[
                    0.18,
                    0.67,
                    0.15,
                ],
            ),
        }
    )

    accounts = assign_account_scenarios(
        rng,
        accounts,
        ScenarioConfig(),
    )

    accounts = attach_behavior_state(
        rng,
        accounts,
    )

    # Coordinated accounts intentionally share infrastructure.
    coordinated = accounts["scenario"].eq(
        "coordinated"
    )

    if coordinated.any():
        coordinated_pool = [
            f"CD{i:04d}"
            for i in range(
                max(
                    1,
                    coordinated.sum() // 3,
                )
            )
        ]

        accounts.loc[
            coordinated,
            "device_id",
        ] = rng.choice(
            coordinated_pool,
            size=coordinated.sum(),
        )

    # Shared infrastructure is deliberately legitimate.
    shared = accounts["scenario"].eq(
        "shared_infrastructure"
    )

    if shared.any():
        shared_pool = [
            f"SH{i:04d}"
            for i in range(
                max(
                    1,
                    shared.sum() // 4,
                )
            )
        ]

        accounts.loc[
            shared,
            "device_id",
        ] = rng.choice(
            shared_pool,
            size=shared.sum(),
        )

    return accounts


def generate_transactions(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    n_transactions: int,
) -> pd.DataFrame:
    """
    Generate a chronological transaction stream.

    Transaction frequency comes from account-level activity state rather than
    uniformly sampling accounts. This gives historical velocity features an
    actual behavioral foundation.
    """

    events = generate_account_event_stream(
        rng=rng,
        accounts=accounts,
        n_transactions=n_transactions,
    )

    account_lookup = (
        accounts
        .set_index("account_id")
    )

    accounts_used = (
        account_lookup
        .loc[events["account_id"]]
        .reset_index()
    )

    timestamps = pd.to_datetime(
        events["timestamp"]
    )

    regime = events[
        "regime"
    ].to_numpy()

    scenario = (
        accounts_used["scenario"]
        .to_numpy()
    )

    baseline = (
        accounts_used["behavior_baseline"]
        .to_numpy(dtype=float)
    )

    # ------------------------------------------------------------------
    # Amount behavior
    # ------------------------------------------------------------------

    amount = (
        baseline
        * rng.lognormal(
            mean=0.0,
            sigma=0.55,
            size=n_transactions,
        )
    )

    # New accounts generally transact at lower amounts.
    amount *= np.where(
        scenario == "new_account",
        0.75,
        1.0,
    )

    # High-value legitimate customers spend more.
    amount *= np.where(
        scenario == "high_value_legitimate",
        1.15,
        1.0,
    )

    # Compromise causes a regime shift.
    post_compromise = regime == "post"

    compromise_multiplier = (
        accounts_used[
            "compromise_multiplier"
        ]
        .to_numpy(dtype=float)
    )

    amount *= np.where(
        post_compromise,
        compromise_multiplier,
        1.0,
    )

    # Burst sessions can contain somewhat larger transactions.
    burst = scenario == "burst"

    amount *= np.where(
        burst,
        rng.uniform(
            1.15,
            2.00,
            n_transactions,
        ),
        1.0,
    )

    amount = np.clip(
        amount,
        20.0,
        25_000.0,
    )

    # ------------------------------------------------------------------
    # Geographic behavior
    # ------------------------------------------------------------------

    ip_country = rng.choice(
        COUNTRIES,
        size=n_transactions,
        p=[
            0.62,
            0.12,
            0.09,
            0.07,
            0.06,
            0.04,
        ],
    )

    shipping_country = (
        accounts_used[
            "home_country"
        ]
        .to_numpy()
        .copy()
    )

    # Normal cross-border activity exists.
    cross_border = (
        rng.random(n_transactions)
        < 0.08
    )

    if cross_border.any():
        shipping_country[
            cross_border
        ] = rng.choice(
            COUNTRIES,
            size=cross_border.sum(),
        )

    # Compromised accounts become more likely to transact from a foreign
    # location after the behavioral transition.
    compromise_location = (
        post_compromise
        & (
            rng.random(n_transactions)
            < 0.62
        )
    )

    if compromise_location.any():
        foreign = rng.choice(
            COUNTRIES,
            size=compromise_location.sum(),
        )

        home = shipping_country[
            compromise_location
        ]

        # Avoid accidentally selecting the same country.
        foreign = np.where(
            foreign == home,
            "US",
            foreign,
        )

        ip_country[
            compromise_location
        ] = foreign

    # ------------------------------------------------------------------
    # Payment and merchant behavior
    # ------------------------------------------------------------------

    payment_method = rng.choice(
        PAYMENT_METHODS,
        size=n_transactions,
        p=[
            0.48,
            0.32,
            0.12,
            0.08,
        ],
    )

    merchant_category = rng.choice(
        MERCHANT_CATEGORIES,
        size=n_transactions,
        p=[
            0.27,
            0.20,
            0.24,
            0.12,
            0.17,
        ],
    )

    merchant_ids = rng.integers(
        0,
        N_MERCHANTS,
        size=n_transactions,
    )

    # ------------------------------------------------------------------
    # Observable behavioral signals used by the simulator
    # ------------------------------------------------------------------

    hour = (
        pd.Series(timestamps)
        .dt.hour
        .to_numpy()
    )

    night = (
        (hour <= 5)
        | (hour >= 23)
    ).astype(int)

    account_age_days = (
        (
            pd.Series(timestamps)
            .dt.normalize()
            .to_numpy()
            - accounts_used[
                "created_at"
            ]
            .dt.normalize()
            .to_numpy()
        )
        / np.timedelta64(1, "D")
    )

    account_age_days = np.maximum(
        account_age_days.astype(float),
        0.0,
    )

    location_mismatch = (
        ip_country != shipping_country
    ).astype(int)

    amount_log = np.log1p(
        amount
    )

    high_amount = (
        amount > 650
    ).astype(int)

    very_new = (
        account_age_days < 14
    ).astype(int)

    # ------------------------------------------------------------------
    # Infrastructure behavior
    # ------------------------------------------------------------------

    shared_device = (
        pd.Series(
            accounts_used[
                "device_id"
            ]
        )
        .duplicated(
            keep=False
        )
        .to_numpy()
    )

    coordinated_signal = (
        scenario == "coordinated"
    ).astype(int)

    legitimate_shared_signal = (
        scenario
        == "shared_infrastructure"
    ).astype(int)

    # ------------------------------------------------------------------
    # Chargeback probability
    # ------------------------------------------------------------------

    risk_logit = (
        -5.35
        + 0.38
        * (
            amount_log
            - np.log(150.0)
        )
        + 0.80 * high_amount
        + 0.75 * very_new
        + 0.72 * location_mismatch
        + 0.50 * night
        + 0.35
        * (
            payment_method == "card"
        ).astype(int)
        + 0.32
        * shared_device.astype(int)
        + 0.95
        * coordinated_signal
        + 1.55
        * post_compromise.astype(int)
        + 0.85
        * burst.astype(int)
        - 0.55
        * legitimate_shared_signal
    )

    # Irreducible uncertainty prevents deterministic labels.
    risk_logit += rng.normal(
        loc=0.0,
        scale=0.28,
        size=n_transactions,
    )

    chargeback_probability = sigmoid(
        risk_logit
    )

    is_chargeback = (
        rng.random(n_transactions)
        < chargeback_probability
    )

    # ------------------------------------------------------------------
    # Device transition
    # ------------------------------------------------------------------

    device_id = (
        accounts_used[
            "device_id"
        ]
        .to_numpy()
        .copy()
    )

    if post_compromise.any():
        device_id[
            post_compromise
        ] = accounts_used.loc[
            post_compromise,
            "secondary_device_id",
        ].to_numpy()

    # ------------------------------------------------------------------
    # Public transaction dataset
    # ------------------------------------------------------------------

    transactions = pd.DataFrame(
        {
            "transaction_id": [
                f"T{i:08d}"
                for i in range(
                    n_transactions
                )
            ],
            "account_id": accounts_used[
                "account_id"
            ].to_numpy(),
            "merchant_id": [
                f"M{i:04d}"
                for i in merchant_ids
            ],
            "timestamp": timestamps,
            "amount": np.round(
                amount,
                2,
            ),
            "currency": "INR",
            "payment_method": payment_method,
            "merchant_category": merchant_category,
            "ip_country": ip_country,
            "shipping_country": shipping_country,
            "device_id": device_id,
            "is_chargeback": (
                is_chargeback.astype(int)
            ),
        }
    )

    return (
        transactions
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def generate_chargebacks(
    rng: np.random.Generator,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Generate delayed chargeback observations."""

    chargebacks = transactions.loc[
        transactions["is_chargeback"] == 1,
        [
            "transaction_id",
            "timestamp",
        ],
    ].copy()

    chargebacks[
        "chargeback_at"
    ] = (
        chargebacks["timestamp"]
        + pd.to_timedelta(
            rng.integers(
                3,
                35,
                len(chargebacks),
            ),
            unit="D",
        )
    )

    chargebacks[
        "reason_code"
    ] = rng.choice(
        [
            "fraud",
            "card_not_present",
            "duplicate",
            "service_not_received",
        ],
        size=len(chargebacks),
        p=[
            0.52,
            0.24,
            0.08,
            0.16,
        ],
    )

    return chargebacks[
        [
            "transaction_id",
            "chargeback_at",
            "reason_code",
        ]
    ]


def main() -> None:
    """Generate the complete reproducible E2 dataset."""

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    GENERATED_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    accounts = generate_accounts(
        rng,
        n_accounts=5_000,
    )

    transactions = generate_transactions(
        rng,
        accounts,
        n_transactions=50_000,
    )

    chargebacks = generate_chargebacks(
        rng,
        transactions,
    )

    # Fail closed if the synthetic world violates its own invariants.
    validate_account_scenarios(
        accounts
    )

    validate_transactions(
        transactions
    )

    accounts.to_parquet(
        ACCOUNTS_PATH,
        index=False,
    )

    transactions.to_parquet(
        TRANSACTIONS_PATH,
        index=False,
    )

    chargebacks.to_parquet(
        CHARGEBACKS_PATH,
        index=False,
    )

    scenario_counts = (
        accounts["scenario"]
        .value_counts()
        .sort_index()
    )

    print(
        f"accounts={len(accounts):,}"
    )

    print(
        f"transactions={len(transactions):,}"
    )

    print(
        f"chargebacks={len(chargebacks):,}"
    )

    print(
        "chargeback_rate="
        f"{transactions.is_chargeback.mean():.4%}"
    )

    print(
        "scenario_counts="
    )

    for (
        scenario_name,
        count,
    ) in scenario_counts.items():
        print(
            f"  {scenario_name}="
            f"{count:,}"
        )

    print(
        f"written_to={GENERATED_DATA}"
    )


if __name__ == "__main__":
    main()