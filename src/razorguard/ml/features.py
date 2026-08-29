from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


# ============================================================
# POINT-IN-TIME ROLLING FEATURES
# ============================================================


def _prior_rolling_features(
    df: pd.DataFrame,
    entity_col: str,
    windows_minutes: list[int],
) -> pd.DataFrame:
    """
    Calculate rolling count and amount using ONLY events
    strictly before the current transaction.

    The current event is appended to the state only after
    its historical features have been calculated.
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

    result: dict[str, np.ndarray] = {}

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

            # Exclude events at or before the cutoff.
            # Therefore the window is strictly:
            # (timestamp - window, timestamp)
            while queue and queue[0][0] <= cutoff:
                queue.popleft()

            result[f"{entity_col}_prior_count_{window}m"][i] = len(queue)

            result[f"{entity_col}_prior_amount_{window}m"][i] = sum(
                value
                for _, value in queue
            )

        # IMPORTANT:
        # Add current event AFTER calculating its features.
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


# ============================================================
# POINT-IN-TIME UNIQUE ENTITY FEATURES
# ============================================================


def _prior_unique_entities(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> pd.Series:
    """
    Number of unique target entities observed previously
    for each source entity.

    Example:
        account -> merchant

        A -> M1  => 0
        A -> M2  => 1
        A -> M3  => 2
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


def _prior_unique_entities_rolling(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    window_minutes: int,
) -> pd.Series:
    """
    Count unique target entities observed for a source entity
    inside a historical rolling window.

    The current transaction is NEVER included.
    """

    sources = df[source_col].to_numpy()
    targets = df[target_col].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    window = pd.Timedelta(minutes=window_minutes)

    states: dict[object, deque] = defaultdict(deque)
    result = np.zeros(
        len(df),
        dtype=np.int32,
    )

    for i, (source, target, timestamp) in enumerate(
        zip(sources, targets, timestamps)
    ):
        timestamp = pd.Timestamp(timestamp)
        queue = states[source]

        cutoff = timestamp - window

        # Remove events outside the historical window.
        while queue and queue[0][0] <= cutoff:
            queue.popleft()

        # Current event has NOT been inserted yet.
        result[i] = len({value for _, value in queue})

        # Insert current event only after feature calculation.
        queue.append(
            (
                timestamp,
                target,
            )
        )

    return pd.Series(
        result,
        index=df.index,
    )


# ============================================================
# HISTORICAL FEATURES
# ============================================================


def add_historical_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build strictly point-in-time transaction features.

    No future transaction information is used.
    The chargeback label is retained in the returned dataframe
    for evaluation/training but is never part of runtime features.
    """

    required_columns = {
        "transaction_id",
        "account_id",
        "merchant_id",
        "device_id",
        "timestamp",
        "amount",
        "ip_country",
        "shipping_country",
    }

    missing = required_columns - set(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing required transaction columns: {sorted(missing)}"
        )

    df = (
        transactions
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
    )

    # ========================================================
    # 1. ACCOUNT HISTORY
    # ========================================================

    prior_count = (
        df.groupby(
            "account_id",
            sort=False,
        )
        .cumcount()
    )

    prior_amount_sum = (
        df.groupby(
            "account_id",
            sort=False,
        )["amount"]
        .cumsum()
        - df["amount"]
    )

    amount_squared = df["amount"] ** 2

    prior_amount_squared_sum = (
        amount_squared.groupby(
            df["account_id"],
            sort=False,
        )
        .cumsum()
        - amount_squared
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

    fallback_amount = float(
        df["amount"].median()
    )

    df["prior_amount_mean"] = (
        prior_mean
        .fillna(fallback_amount)
    )

    df["prior_amount_std"] = (
        prior_variance
        .fillna(0)
        .pow(0.5)
    )

    # ========================================================
    # 2. BEHAVIORAL DEVIATION
    # ========================================================

    df["amount_vs_history"] = (
        df["amount"]
        /
        df["prior_amount_mean"]
        .replace(
            0,
            fallback_amount,
        )
    )

    df["amount_zscore"] = (
        (
            df["amount"]
            - df["prior_amount_mean"]
        )
        /
        df["prior_amount_std"]
        .replace(
            0,
            np.nan,
        )
    ).fillna(0)

    # ========================================================
    # 3. TIME FEATURES
    # ========================================================

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

    # ========================================================
    # 4. TIME SINCE PREVIOUS TRANSACTION
    # ========================================================

    df["previous_timestamp"] = (
        df.groupby(
            "account_id",
            sort=False,
        )["timestamp"]
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

    df["minutes_since_previous_tx"] = (
        df["minutes_since_previous_tx"]
        .fillna(10_000)
    )

    # ========================================================
    # 5. DORMANCY / RETURN BEHAVIOR
    # ========================================================

    # A return after >= 7 days of inactivity.
    # A long dormancy is >= 30 days.
    #
    # These are based ONLY on the previous transaction,
    # therefore they are safe for real-time inference.

    df["is_dormant_return"] = (
        df["minutes_since_previous_tx"] >= 7 * 24 * 60
    ).astype(int)

    df["is_long_dormancy"] = (
        df["minutes_since_previous_tx"] >= 30 * 24 * 60
    ).astype(int)

    # ========================================================
    # 6. LOCATION
    # ========================================================

    df["location_mismatch"] = (
        df["ip_country"]
        != df["shipping_country"]
    ).astype(int)

    # ========================================================
    # 7. ACCOUNT VELOCITY
    # ========================================================

    account_velocity = _prior_rolling_features(
        df,
        "account_id",
        [5, 60, 1440],
    )

    df = pd.concat(
        [
            df,
            account_velocity,
        ],
        axis=1,
    )

    # ========================================================
    # 8. DEVICE VELOCITY
    # ========================================================

    device_velocity = _prior_rolling_features(
        df,
        "device_id",
        [60, 1440],
    )

    df = pd.concat(
        [
            df,
            device_velocity,
        ],
        axis=1,
    )

    # ========================================================
    # 9. MERCHANT VELOCITY
    # ========================================================

    merchant_velocity = _prior_rolling_features(
        df,
        "merchant_id",
        [60, 1440],
    )

    df = pd.concat(
        [
            df,
            merchant_velocity,
        ],
        axis=1,
    )

    # ========================================================
    # 10. ACCOUNT -> MERCHANT HISTORY
    # ========================================================

    df["prior_unique_merchants"] = (
        _prior_unique_entities(
            df,
            "account_id",
            "merchant_id",
        )
    )

    df[
        "account_id_prior_unique_merchant_id_60m"
    ] = (
        _prior_unique_entities_rolling(
            df,
            "account_id",
            "merchant_id",
            60,
        )
    )

    # ========================================================
    # 11. DEVICE REUSE
    # ========================================================

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

    # ========================================================
    # 12. DERIVED VELOCITY FEATURES
    # ========================================================

    df["account_amount_per_tx_1h"] = (
        df["account_id_prior_amount_60m"]
        /
        df["account_id_prior_count_60m"]
        .replace(
            0,
            np.nan,
        )
    ).fillna(0)

    df["account_velocity_ratio"] = (
        df["account_id_prior_count_5m"]
        /
        (
            df["account_id_prior_count_60m"]
            + 1
        )
    )

    # ========================================================
    # 13. RUNTIME SAFETY CHECK
    # ========================================================

    runtime_columns = (
        set(df.columns)
        -
        {
            "is_chargeback",
            "transaction_id",
            "timestamp",
            "previous_timestamp",
        }
    )

    if "is_chargeback" in runtime_columns:
        raise RuntimeError(
            "LABEL LEAKAGE: is_chargeback entered features"
        )

    return df


# ============================================================
# MODEL FEATURE DEFINITIONS
# ============================================================


NUMERIC_FEATURES = [
    # Transaction
    "amount",

    # Account history
    "prior_tx_count",
    "prior_amount_mean",
    "prior_amount_std",

    # Behavioral deviation
    "amount_vs_history",
    "amount_zscore",

    # Time
    "hour",
    "day_of_week",
    "is_night",
    "minutes_since_previous_tx",

    # Dormancy
    "is_dormant_return",
    "is_long_dormancy",

    # Location
    "location_mismatch",

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

    # Account -> merchant behavior
    "prior_unique_merchants",
    "account_id_prior_unique_merchant_id_60m",

    # Device reuse
    "prior_accounts_per_device",
    "device_accounts_signal",

    # Derived velocity
    "account_amount_per_tx_1h",
    "account_velocity_ratio",
]


CATEGORICAL_FEATURES = [
    "payment_method",
    "merchant_category",
    "ip_country",
    "shipping_country",
]


# ============================================================
# MODEL FRAME
# ============================================================


def build_model_frame(
    transactions: pd.DataFrame,
    *,
    include_label: bool | None = None,
) -> pd.DataFrame:
    """
    Build the final point-in-time model frame.

    Training/evaluation data includes ``is_chargeback``.
    Runtime inference data does not require the label.

    When ``include_label`` is None, the label is included only
    when it exists in the input dataframe.
    """

    df = add_historical_features(
        transactions,
    )

    columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "timestamp",
            "transaction_id",
        ]
    )

    if include_label is None:
        include_label = "is_chargeback" in df.columns

    if include_label:
        if "is_chargeback" not in df.columns:
            raise ValueError(
                "include_label=True requires "
                "'is_chargeback' in the input data"
            )

        columns.append("is_chargeback")

    return df[columns].copy()