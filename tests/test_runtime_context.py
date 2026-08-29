from __future__ import annotations

import pandas as pd
import pytest

from razorguard.runtime.context import (
    RuntimeContextStore,
)


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
                "T3",
                "T4",
            ],
            "account_id": [
                "A1",
                "A1",
                "A2",
                "A1",
            ],
            "merchant_id": [
                "M1",
                "M2",
                "M1",
                "M3",
            ],
            "device_id": [
                "D1",
                "D1",
                "D2",
                "D1",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:05:00",
                    "2026-01-01 10:10:00",
                    "2026-01-01 10:20:00",
                ]
            ),
            "amount": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "ip_country": [
                "IN",
                "IN",
                "US",
                "IN",
            ],
            "shipping_country": [
                "IN",
                "IN",
                "US",
                "IN",
            ],
            "payment_method": [
                "upi",
                "upi",
                "card",
                "upi",
            ],
            "merchant_category": [
                "food",
                "food",
                "electronics",
                "travel",
            ],
            "is_chargeback": [
                0,
                0,
                1,
                0,
            ],
        }
    )


def write_history(
    tmp_path,
) -> str:
    path = (
        tmp_path
        / "transactions.parquet"
    )

    transactions().to_parquet(
        path,
        index=False,
    )

    return str(path)


def test_store_loads_history(tmp_path):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    assert len(store.transactions) == 4


def test_history_is_strictly_before_timestamp(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    history = store.get_transaction_history(
        "2026-01-01 10:10:00"
    )

    assert list(
        history["transaction_id"]
    ) == [
        "T1",
        "T2",
    ]


def test_account_history_filters_correctly(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    history = store.get_account_history(
        "A1",
        "2026-01-01 10:20:00",
    )

    assert list(
        history["transaction_id"]
    ) == [
        "T1",
        "T2",
    ]


def test_device_history_filters_correctly(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    history = store.get_device_history(
        "D1",
        "2026-01-01 10:20:00",
    )

    assert list(
        history["transaction_id"]
    ) == [
        "T1",
        "T2",
    ]


def test_merchant_history_filters_correctly(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    history = store.get_merchant_history(
        "M1",
        "2026-01-01 10:20:00",
    )

    assert list(
        history["transaction_id"]
    ) == [
        "T1",
        "T3",
    ]


def test_current_transaction_is_not_in_history(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    history = store.get_account_history(
        "A1",
        "2026-01-01 10:20:00",
    )

    assert "T4" not in set(
        history["transaction_id"]
    )


def test_build_scoring_frame_appends_current_transaction(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    current = {
        "transaction_id": "LIVE-1",
        "account_id": "A1",
        "merchant_id": "M9",
        "device_id": "D9",
        "timestamp": "2026-01-01 10:30:00",
        "amount": 900.0,
        "ip_country": "IN",
        "shipping_country": "IN",
        "payment_method": "card",
        "merchant_category": "electronics",
    }

    frame = store.build_scoring_frame(
        current
    )

    assert frame.iloc[-1][
        "transaction_id"
    ] == "LIVE-1"

    assert len(frame) == 5


def test_build_scoring_frame_excludes_future_transactions(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    current = {
        "transaction_id": "LIVE-1",
        "account_id": "A1",
        "merchant_id": "M9",
        "device_id": "D9",
        "timestamp": "2026-01-01 10:15:00",
        "amount": 900.0,
        "ip_country": "IN",
        "shipping_country": "IN",
        "payment_method": "card",
        "merchant_category": "electronics",
    }

    frame = store.build_scoring_frame(
        current
    )

    assert list(
        frame["transaction_id"]
    ) == [
        "T1",
        "T2",
        "T3",
        "LIVE-1",
    ]


def test_context_summary(
    tmp_path,
):
    store = RuntimeContextStore(
        write_history(tmp_path)
    )

    current = {
        "transaction_id": "LIVE-1",
        "account_id": "A1",
        "merchant_id": "M2",
        "device_id": "D1",
        "timestamp": "2026-01-01 10:20:00",
        "amount": 900.0,
        "ip_country": "IN",
        "shipping_country": "IN",
        "payment_method": "card",
        "merchant_category": "electronics",
    }

    summary = store.get_context_summary(
        current
    )

    assert summary[
        "prior_transactions"
    ] == 3

    assert summary[
        "prior_account_transactions"
    ] == 2

    assert summary[
        "prior_device_transactions"
    ] == 2

    assert summary[
        "prior_merchant_transactions"
    ] == 1


def test_missing_history_file_raises(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        RuntimeContextStore(
            tmp_path
            / "missing.parquet"
        )


def test_missing_required_columns_raise(
    tmp_path,
):
    path = (
        tmp_path
        / "bad.parquet"
    )

    pd.DataFrame(
        {
            "transaction_id": ["T1"],
        }
    ).to_parquet(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        RuntimeContextStore(path)