from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def _prior_rolling_features(
    df: pd.DataFrame,
    entity_col: str,
    windows_minutes: list[int],
) -> pd.DataFrame:
    """
    Calculate rolling count and amount using ONLY events before
    the current transaction.

    Uses pandas Timedelta objects rather than relying on the
    internal integer representation of datetime64.
    """

    entities = df[entity_col].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    amounts = df["amount"].to_numpy(dtype=float)

    states = defaultdict(
        lambda: {
            window: deque()
            for window in windows_minutes
        }
    )

    result = {}

    for window in windows_minutes:
        result[f"{entity_col}_prior_count_{window}m"] = np.zeros(
            len(df),
            dtype=np.int32,
        )

        result[f"{entity_col}_prior_amount_{window}m"] = np.zeros(
            len(df),
            dtype=float,
        )

    window_deltas = {
        window: pd.Timedelta(minutes=window)
        for window in windows_minutes
    }

    for i, (entity, timestamp, amount) in enumerate(
        zip(entities, timestamps, amounts)
    ):
        timestamp = pd.Timestamp(timestamp)

        for window in windows_minutes:

            queue = states[entity][window]

            cutoff = timestamp - window_deltas[window]

            while queue and queue[0][0] <= cutoff:
                queue.popleft()

            result[
                f"{entity_col}_prior_count_{window}m"
            ][i] = len(queue)

            result[
                f"{entity_col}_prior_amount_{window}m"
            ][i] = sum(
                value
                for _, value in queue
            )

        # IMPORTANT:
        #
        # The current transaction is added AFTER
        # calculating its features.
        #
        # Therefore it can NEVER leak into its
        # own velocity features.
        for window in windows_minutes:
            states[entity][window].append(
                (
                    timestamp,
                    amount,
                )
            )

    return pd.DataFrame(
        result,
        index=df.index,
    )


def _prior_unique_entities(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> pd.Series:
    """
    Number of unique target entities observed previously
    for each source entity.
    """

    seen = defaultdict(set)

    result = np.zeros(
        len(df),
        dtype=np.int32,
    )

    sources = df[source_col].to_numpy()
    targets = df[target_col].to_numpy()

    for i, (source, target) in enumerate(
        zip(sources, targets)
    ):
        result[i] = len(seen[source])

        seen[source].add(target)

    return pd.Series(
        result,
        index=df.index,
    )


def add_historical_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    df = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # ============================================================
    # 1. ACCOUNT HISTORY
    # ============================================================

    prior_count = (
        df.groupby("account_id")
        .cumcount()
    )

    prior_amount_sum = (
        df.groupby("account_id")["amount"]
        .cumsum()
        - df["amount"]
    )

    prior_amount_squared_sum = (
        df.assign(
            amount_squared=df["amount"] ** 2
        )
        .groupby("account_id")["amount_squared"]
        .cumsum()
        - df["amount"] ** 2
    )

    count_float = (
        prior_count
        .astype(float)
        .replace(0, np.nan)
    )

    prior_mean = (
        prior_amount_sum
        / count_float
    )

    prior_second_moment = (
        prior_amount_squared_sum
        / count_float
    )

    prior_variance = (
        prior_second_moment
        - prior_mean ** 2
    ).clip(lower=0)

    df["prior_tx_count"] = prior_count

    df["prior_amount_mean"] = (
        prior_mean
        .fillna(df["amount"].median())
    )

    df["prior_amount_std"] = (
        prior_variance
        .fillna(0)
        .pow(0.5)
    )

    # ============================================================
    # 2. BEHAVIORAL DEVIATION
    # ============================================================

    df["amount_vs_history"] = (
        df["amount"]
        /
        df["prior_amount_mean"]
        .replace(
            0,
            df["amount"].median(),
        )
    )

    df["amount_zscore"] = (
        (
            df["amount"]
            - df["prior_amount_mean"]
        )
        /
        df["prior_amount_std"]
        .replace(0, np.nan)
    ).fillna(0)

    # ============================================================
    # 3. TIME FEATURES
    # ============================================================

    df["hour"] = (
        df["timestamp"]
        .dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"]
        .dt.dayofweek
    )

    df["is_night"] = (
        (df["hour"] <= 5)
        |
        (df["hour"] >= 23)
    ).astype(int)

    # ============================================================
    # 4. TIME SINCE PREVIOUS TRANSACTION
    # ============================================================

    df["previous_timestamp"] = (
        df.groupby("account_id")["timestamp"]
        .shift(1)
    )

    df["minutes_since_previous_tx"] = (
        (
            df["timestamp"]
            - df["previous_timestamp"]
        )
        .dt.total_seconds()
        / 60.0
    )

    # No previous transaction = very large gap.
    df["minutes_since_previous_tx"] = (
        df["minutes_since_previous_tx"]
        .fillna(10_000)
    )

    # ============================================================
    # 5. LOCATION
    # ============================================================

    df["location_mismatch"] = (
        df["ip_country"]
        != df["shipping_country"]
    ).astype(int)

    # ============================================================
    # 6. ACCOUNT VELOCITY
    # ============================================================

    account_velocity = _prior_rolling_features(
        df,
        "account_id",
        [5, 60, 1440],
    )

    df = pd.concat(
        [df, account_velocity],
        axis=1,
    )

    # ============================================================
    # 7. DEVICE VELOCITY
    # ============================================================

    device_velocity = _prior_rolling_features(
        df,
        "device_id",
        [60, 1440],
    )

    df = pd.concat(
        [df, device_velocity],
        axis=1,
    )

    # ============================================================
    # 8. MERCHANT VELOCITY
    # ============================================================

    merchant_velocity = _prior_rolling_features(
        df,
        "merchant_id",
        [60, 1440],
    )

    df = pd.concat(
        [df, merchant_velocity],
        axis=1,
    )

    # ============================================================
    # 9. DEVICE REUSE
    # ============================================================

    df["prior_accounts_per_device"] = (
        _prior_unique_entities(
            df,
            "device_id",
            "account_id",
        )
    )

    df["device_accounts_signal"] = (
        df["prior_accounts_per_device"] >= 3
    ).astype(int)

    # ============================================================
    # 10. DERIVED VELOCITY FEATURES
    # ============================================================

    df["account_amount_per_tx_1h"] = (
        df["account_id_prior_amount_60m"]
        /
        df["account_id_prior_count_60m"]
        .replace(0, np.nan)
    ).fillna(0)

    df["account_velocity_ratio"] = (
        df["account_id_prior_count_5m"]
        /
        (
            df["account_id_prior_count_60m"]
            + 1
        )
    )

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    # Chargeback is a future outcome and must NEVER be used
    # as a runtime feature.
    runtime_columns = set(
        df.columns
    ) - {
        "is_chargeback",
        "transaction_id",
        "timestamp",
        "previous_timestamp",
    }

    if "is_chargeback" in runtime_columns:
        raise RuntimeError(
            "LABEL LEAKAGE: is_chargeback entered features"
        )

    return df


NUMERIC_FEATURES = [
    "amount",
    "prior_tx_count",
    "prior_amount_mean",
    "prior_amount_std",
    "amount_vs_history",
    "amount_zscore",
    "location_mismatch",
    "hour",
    "day_of_week",
    "is_night",
    "minutes_since_previous_tx",

    # Account velocity
    "account_id_prior_count_5m",
    "account_id_prior_amount_5m",
    "account_id_prior_count_60m",
    "account_id_prior_amount_60m",
    "account_id_prior_count_1440m",
    "account_id_prior_amount_1440m",

    # Device velocity
    "device_id_prior_count_60m",
    "device_id_prior_amount_60m",
    "device_id_prior_count_1440m",
    "device_id_prior_amount_1440m",

    # Merchant velocity
    "merchant_id_prior_count_60m",
    "merchant_id_prior_amount_60m",
    "merchant_id_prior_count_1440m",
    "merchant_id_prior_amount_1440m",

    # Network/behavior
    "prior_accounts_per_device",
    "device_accounts_signal",
    "account_amount_per_tx_1h",
    "account_velocity_ratio",
]


CATEGORICAL_FEATURES = [
    "payment_method",
    "merchant_category",
    "ip_country",
    "shipping_country",
]


def build_model_frame(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    df = add_historical_features(
        transactions
    )

    return df[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "timestamp",
            "transaction_id",
            "is_chargeback",
        ]
    ].copy()