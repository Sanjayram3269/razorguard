from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RiskCluster:
    """A deterministic, explainable coordinated-risk cluster."""

    cluster_id: str
    cluster_type: str
    risk_score: float

    accounts: list[str]
    devices: list[str]
    merchants: list[str]
    transactions: list[str]

    signals: list[dict[str, Any]]
    evidence: list[str]
    timeline: list[dict[str, Any]]


def _normalise(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def _safe_score(value: float) -> float:
    return round(
        max(
            0.0,
            min(
                float(value),
                100.0,
            ),
        ),
        2,
    )


def _validate_columns(
    transactions: pd.DataFrame,
) -> None:
    required = {
        "account_id",
        "device_id",
        "merchant_id",
        "timestamp",
    }

    missing = required - set(
        transactions.columns,
    )

    if missing:
        raise ValueError(
            "Missing required transaction columns: "
            f"{sorted(missing)}",
        )


def _prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    _validate_columns(transactions)

    frame = transactions.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["timestamp"],
    )

    frame = frame.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    return frame


def _transaction_id(
    row: pd.Series,
    fallback: int,
) -> str:
    if "transaction_id" in row.index:
        value = row["transaction_id"]

        if pd.notna(value):
            return _normalise(value)

    return f"ROW-{fallback:08d}"


def find_connected_accounts(
    transactions: pd.DataFrame,
    account_id: str,
    device_id: str,
    merchant_id: str,
) -> set[str]:
    """
    Find accounts connected to the target account through
    the target device or merchant.

    This function is deterministic and uses only observed
    transaction relationships.
    """

    frame = _prepare_transactions(
        transactions,
    )

    accounts: set[str] = {
        _normalise(account_id),
    }

    device_matches = frame[
        frame["device_id"].astype(str)
        == _normalise(device_id)
    ]

    merchant_matches = frame[
        frame["merchant_id"].astype(str)
        == _normalise(merchant_id)
    ]

    accounts.update(
        device_matches["account_id"]
        .astype(str)
        .tolist(),
    )

    accounts.update(
        merchant_matches["account_id"]
        .astype(str)
        .tolist(),
    )

    return accounts


def find_shared_entities(
    transactions: pd.DataFrame,
    accounts: set[str],
) -> dict[str, set[str]]:
    """
    Return the devices and merchants associated with a group
    of accounts.
    """

    frame = _prepare_transactions(
        transactions,
    )

    account_values = {
        _normalise(account)
        for account in accounts
    }

    subset = frame[
        frame["account_id"].astype(str).isin(
            account_values,
        )
    ]

    return {
        "devices": set(
            subset["device_id"]
            .astype(str)
            .tolist(),
        ),
        "merchants": set(
            subset["merchant_id"]
            .astype(str)
            .tolist(),
        ),
    }


def detect_temporal_bursts(
    transactions: pd.DataFrame,
    accounts: set[str],
    window_minutes: int = 10,
    minimum_transactions: int = 3,
) -> list[dict[str, Any]]:
    """
    Detect bursts where multiple transactions involving the
    cluster occur within a short time window.
    """

    frame = _prepare_transactions(
        transactions,
    )

    subset = frame[
        frame["account_id"].astype(str).isin(
            {
                _normalise(account)
                for account in accounts
            },
        )
    ].copy()

    if len(subset) < minimum_transactions:
        return []

    window = pd.Timedelta(
        minutes=window_minutes,
    )

    bursts: list[dict[str, Any]] = []

    timestamps = subset["timestamp"].tolist()

    start = 0

    for end in range(
        len(timestamps),
    ):
        while (
            timestamps[end]
            - timestamps[start]
            > window
        ):
            start += 1

        count = end - start + 1

        if count >= minimum_transactions:
            burst = subset.iloc[
                start : end + 1
            ]

            bursts.append(
                {
                    "start": burst["timestamp"]
                    .iloc[0]
                    .isoformat(),
                    "end": burst["timestamp"]
                    .iloc[-1]
                    .isoformat(),
                    "transaction_count": int(
                        len(burst),
                    ),
                    "accounts": sorted(
                        set(
                            burst["account_id"]
                            .astype(str)
                            .tolist(),
                        ),
                    ),
                },
            )

    # Keep only distinct windows.
    unique: dict[
        tuple[str, str, int],
        dict[str, Any],
    ] = {}

    for burst in bursts:
        key = (
            burst["start"],
            burst["end"],
            burst["transaction_count"],
        )

        unique[key] = burst

    return list(
        unique.values(),
    )


def calculate_cluster_risk(
    *,
    account_count: int,
    shared_device_count: int,
    shared_merchant_count: int,
    transaction_count: int,
    burst_count: int,
) -> float:
    """
    Calculate an explainable coordination score.

    This is intentionally deterministic. It is NOT an ML model.
    """

    score = 0.0

    # Multiple connected accounts.
    score += min(
        max(account_count - 1, 0) * 8.0,
        24.0,
    )

    # Shared devices are stronger coordination signals.
    score += min(
        shared_device_count * 22.0,
        30.0,
    )

    # Shared merchants provide supporting evidence.
    score += min(
        shared_merchant_count * 8.0,
        16.0,
    )

    # Larger related transaction groups add evidence.
    score += min(
        max(transaction_count - 1, 0) * 1.5,
        15.0,
    )

    # Temporal bursts are strong coordination evidence.
    score += min(
        burst_count * 10.0,
        15.0,
    )

    return _safe_score(score)


def build_cluster_evidence(
    *,
    accounts: set[str],
    devices: set[str],
    merchants: set[str],
    transactions: pd.DataFrame,
    bursts: list[dict[str, Any]],
) -> list[str]:
    evidence: list[str] = []

    if len(accounts) > 1:
        evidence.append(
            f"{len(accounts)} accounts are connected "
            "through observed transaction relationships.",
        )

    if len(devices) > 1:
        evidence.append(
            f"{len(devices)} devices are associated "
            "with the connected account group.",
        )

    if len(merchants) > 1:
        evidence.append(
            f"{len(merchants)} merchants are associated "
            "with the connected account group.",
        )

    if bursts:
        evidence.append(
            f"{len(bursts)} temporal transaction burst(s) "
            "were detected.",
        )

    if len(transactions) >= 3:
        evidence.append(
            f"{len(transactions)} related transactions "
            "were observed.",
        )

    return evidence


def build_cluster_signals(
    *,
    accounts: set[str],
    devices: set[str],
    merchants: set[str],
    transaction_count: int,
    bursts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    if len(accounts) > 1:
        signals.append(
            {
                "type": "MULTI_ACCOUNT_CONNECTION",
                "severity": "HIGH"
                if len(accounts) >= 3
                else "MEDIUM",
                "value": len(accounts),
                "evidence": (
                    f"{len(accounts)} connected accounts"
                ),
            },
        )

    if len(devices) == 1 and len(accounts) > 1:
        signals.append(
            {
                "type": "SHARED_DEVICE",
                "severity": "HIGH",
                "value": len(accounts),
                "evidence": (
                    f"{len(accounts)} accounts use "
                    f"device {next(iter(devices))}"
                ),
            },
        )

    if len(merchants) == 1 and len(accounts) > 1:
        signals.append(
            {
                "type": "SHARED_MERCHANT",
                "severity": "MEDIUM",
                "value": len(accounts),
                "evidence": (
                    f"{len(accounts)} accounts transact "
                    f"with merchant {next(iter(merchants))}"
                ),
            },
        )

    if transaction_count >= 3:
        signals.append(
            {
                "type": "TRANSACTION_CLUSTER",
                "severity": "MEDIUM",
                "value": transaction_count,
                "evidence": (
                    f"{transaction_count} related transactions"
                ),
            },
        )

    if bursts:
        signals.append(
            {
                "type": "TEMPORAL_BURST",
                "severity": "HIGH",
                "value": len(bursts),
                "evidence": (
                    f"{len(bursts)} transaction burst(s) "
                    "detected in a short time window"
                ),
            },
        )

    return signals


def build_risk_cluster(
    transactions: pd.DataFrame,
    *,
    account_id: str,
    device_id: str,
    merchant_id: str,
    cluster_id: str = "FR-000001",
) -> RiskCluster:
    """
    Build a deterministic coordinated-risk cluster around
    one investigated transaction.
    """

    frame = _prepare_transactions(
        transactions,
    )

    accounts = find_connected_accounts(
        frame,
        account_id=account_id,
        device_id=device_id,
        merchant_id=merchant_id,
    )

    entities = find_shared_entities(
        frame,
        accounts,
    )

    devices = entities["devices"]
    merchants = entities["merchants"]

    related = frame[
        frame["account_id"].astype(str).isin(
            {
                _normalise(account)
                for account in accounts
            },
        )
    ].copy()

    bursts = detect_temporal_bursts(
        related,
        accounts,
    )

    transaction_ids = [
        _transaction_id(
            row,
            index,
        )
        for index, (_, row)
        in enumerate(
            related.iterrows(),
        )
    ]

    signals = build_cluster_signals(
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        transaction_count=len(related),
        bursts=bursts,
    )

    evidence = build_cluster_evidence(
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        transactions=related,
        bursts=bursts,
    )

    risk_score = calculate_cluster_risk(
        account_count=len(accounts),
        shared_device_count=(
            1
            if len(devices) == 1 and len(accounts) > 1
            else 0
        ),
        shared_merchant_count=(
            1
            if len(merchants) == 1 and len(accounts) > 1
            else 0
        ),
        transaction_count=len(related),
        burst_count=len(bursts),
    )

    timeline = [
        {
            "transaction_id": transaction_id,
            "timestamp": row["timestamp"].isoformat(),
            "account_id": _normalise(
                row["account_id"],
            ),
            "device_id": _normalise(
                row["device_id"],
            ),
            "merchant_id": _normalise(
                row["merchant_id"],
            ),
        }
        for transaction_id, (_, row)
        in zip(
            transaction_ids,
            related.iterrows(),
        )
    ]

    cluster_type = (
        "COORDINATED_NETWORK"
        if (
            len(accounts) > 1
            and (
                len(devices) > 1
                or len(merchants) > 1
                or bursts
            )
        )
        else "CONNECTED_ACTIVITY"
    )

    return RiskCluster(
        cluster_id=cluster_id,
        cluster_type=cluster_type,
        risk_score=risk_score,
        accounts=sorted(accounts),
        devices=sorted(devices),
        merchants=sorted(merchants),
        transactions=transaction_ids,
        signals=signals,
        evidence=evidence,
        timeline=timeline,
    )