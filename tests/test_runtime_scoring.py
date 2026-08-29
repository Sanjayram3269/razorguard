from __future__ import annotations

import pandas as pd

from razorguard.runner.batch import (
    score_runtime_transaction,
)
from razorguard.runtime.context import (
    RuntimeContextStore,
)


def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
            ],
            "account_id": [
                "A1",
                "A1",
            ],
            "merchant_id": [
                "M1",
                "M2",
            ],
            "device_id": [
                "D1",
                "D1",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:05:00",
                ]
            ),
            "amount": [
                100.0,
                200.0,
            ],
            "ip_country": [
                "IN",
                "IN",
            ],
            "shipping_country": [
                "IN",
                "IN",
            ],
            "payment_method": [
                "upi",
                "upi",
            ],
            "merchant_category": [
                "food",
                "food",
            ],
            "is_chargeback": [
                0,
                0,
            ],
        }
    )


def current_transaction() -> dict:
    return {
        "transaction_id": "LIVE-1",
        "account_id": "A1",
        "merchant_id": "M3",
        "device_id": "D1",
        "timestamp": "2026-01-01 10:10:00",
        "amount": 5000.0,
        "ip_country": "US",
        "shipping_country": "IN",
        "payment_method": "card",
        "merchant_category": "electronics",
    }


class FakeModel:
    def predict_proba(self, X):
        return pd.DataFrame(
            {
                0: [0.1] * len(X),
                1: [0.9] * len(X),
            }
        ).to_numpy()


def test_runtime_scoring_returns_current_transaction(
    tmp_path,
):
    path = (
        tmp_path
        / "transactions.parquet"
    )

    history().to_parquet(
        path,
        index=False,
    )

    store = RuntimeContextStore(
        path
    )

    result = score_runtime_transaction(
        current_transaction(),
        FakeModel(),
        store,
    )

    assert result[
        "transaction_id"
    ] == "LIVE-1"

    assert "risk_score" in result
    assert "risk_level" in result
    assert "decision" in result
    assert "evidence_text" in result


def test_runtime_scoring_uses_historical_context(
    tmp_path,
):
    path = (
        tmp_path
        / "transactions.parquet"
    )

    history().to_parquet(
        path,
        index=False,
    )

    store = RuntimeContextStore(
        path
    )

    frame = store.build_scoring_frame(
        current_transaction()
    )

    assert list(
        frame["transaction_id"]
    ) == [
        "T1",
        "T2",
        "LIVE-1",
    ]


def test_runtime_scoring_does_not_use_future_events(
    tmp_path,
):
    data = history()

    data = pd.concat(
        [
            data,
            pd.DataFrame(
                [
                    {
                        "transaction_id": "FUTURE",
                        "account_id": "A1",
                        "merchant_id": "M99",
                        "device_id": "D99",
                        "timestamp": pd.Timestamp(
                            "2026-01-01 10:20:00"
                        ),
                        "amount": 999999.0,
                        "ip_country": "RU",
                        "shipping_country": "US",
                        "payment_method": "card",
                        "merchant_category": "unknown",
                        "is_chargeback": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    path = (
        tmp_path
        / "transactions.parquet"
    )

    data.to_parquet(
        path,
        index=False,
    )

    store = RuntimeContextStore(
        path
    )

    frame = store.build_scoring_frame(
        current_transaction()
    )

    assert "FUTURE" not in set(
        frame["transaction_id"]
    )


def test_runtime_scoring_preserves_case_artifact(
    tmp_path,
):
    path = (
        tmp_path
        / "transactions.parquet"
    )

    history().to_parquet(
        path,
        index=False,
    )

    store = RuntimeContextStore(
        path
    )

    result = score_runtime_transaction(
        current_transaction(),
        FakeModel(),
        store,
    )

    assert "case" in result

    assert (
        result["case"]["transaction_id"]
        == "LIVE-1"
    )