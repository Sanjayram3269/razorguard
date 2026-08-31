from __future__ import annotations

import pandas as pd

from razorguard.graph.clusters import (
    build_risk_cluster,
    calculate_cluster_risk,
    detect_temporal_bursts,
    find_connected_accounts,
)


def make_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transaction_id": "T001",
                "timestamp": "2026-01-01 10:00:00",
                "account_id": "A001",
                "device_id": "D001",
                "merchant_id": "M001",
            },
            {
                "transaction_id": "T002",
                "timestamp": "2026-01-01 10:02:00",
                "account_id": "A002",
                "device_id": "D001",
                "merchant_id": "M001",
            },
            {
                "transaction_id": "T003",
                "timestamp": "2026-01-01 10:04:00",
                "account_id": "A003",
                "device_id": "D001",
                "merchant_id": "M001",
            },
            {
                "transaction_id": "T004",
                "timestamp": "2026-01-01 10:20:00",
                "account_id": "A004",
                "device_id": "D004",
                "merchant_id": "M004",
            },
        ],
    )


def test_find_connected_accounts() -> None:
    frame = make_transactions()

    accounts = find_connected_accounts(
        frame,
        account_id="A001",
        device_id="D001",
        merchant_id="M001",
    )

    assert accounts == {
        "A001",
        "A002",
        "A003",
    }


def test_temporal_burst_detection() -> None:
    frame = make_transactions()

    bursts = detect_temporal_bursts(
        frame,
        {
            "A001",
            "A002",
            "A003",
        },
        window_minutes=10,
        minimum_transactions=3,
    )

    assert bursts
    assert bursts[0]["transaction_count"] >= 3


def test_temporal_burst_ignores_distant_transaction() -> None:
    frame = make_transactions()

    bursts = detect_temporal_bursts(
        frame,
        {
            "A001",
            "A002",
            "A003",
        },
        window_minutes=10,
        minimum_transactions=3,
    )

    for burst in bursts:
        assert "A004" not in burst["accounts"]


def test_cluster_risk_is_bounded() -> None:
    score = calculate_cluster_risk(
        account_count=10,
        shared_device_count=4,
        shared_merchant_count=3,
        transaction_count=100,
        burst_count=5,
    )

    assert 0.0 <= score <= 100.0


def test_cluster_builds_coordinated_network() -> None:
    frame = make_transactions()

    cluster = build_risk_cluster(
        frame,
        account_id="A001",
        device_id="D001",
        merchant_id="M001",
    )

    assert cluster.cluster_type == "COORDINATED_NETWORK"

    assert {
        "A001",
        "A002",
        "A003",
    }.issubset(
        set(cluster.accounts),
    )

    assert "D001" in cluster.devices
    assert "M001" in cluster.merchants

    assert cluster.transactions
    assert cluster.signals
    assert cluster.evidence
    assert cluster.timeline

    assert 0.0 <= cluster.risk_score <= 100.0