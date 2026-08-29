from __future__ import annotations

import json

import pandas as pd
from sklearn.dummy import DummyClassifier

from razorguard.runner.batch import (
    behavioral_signal,
    run_batch,
    score_batch,
)


def test_behavioral_signal_is_bounded():
    row = pd.Series(
        {
            "amount_zscore": 10.0,
            "is_dormant_return": 1,
            "account_id_prior_count_60m": 100,
            "location_mismatch": 1,
            "account_velocity_ratio": 10.0,
        }
    )

    value = behavioral_signal(row)

    assert 0.0 <= value <= 1.0


def test_behavioral_signal_is_zero_for_clean_history():
    row = pd.Series(
        {
            "amount_zscore": 0.0,
            "is_dormant_return": 0,
            "account_id_prior_count_60m": 0,
            "location_mismatch": 0,
            "account_velocity_ratio": 0.0,
        }
    )

    assert behavioral_signal(row) == 0.0


def test_score_batch_produces_cases():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2"],
            "account_id": ["A1", "A1"],
            "merchant_id": ["M1", "M2"],
            "device_id": ["D1", "D1"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:05:00",
                ]
            ),
            "amount": [100.0, 5000.0],
            "ip_country": ["IN", "US"],
            "shipping_country": ["IN", "IN"],
            "payment_method": ["upi", "card"],
            "merchant_category": [
                "food",
                "electronics",
            ],
            "is_chargeback": [0, 1],
        }
    )

    model = DummyClassifier(
        strategy="prior"
    )

    model.fit(
        pd.DataFrame(
            {
                "x": [0, 1],
            }
        ),
        [0, 1],
    )

    # Dummy model requires the same feature columns
    # as the actual pipeline, so use a small wrapper.
    class FakeModel:
        def predict_proba(self, X):
            return pd.DataFrame(
                {
                0: [0.0] * len(X),
                1: [1.0] * len(X),
            }
        ).to_numpy()

    scored, queue = score_batch(
        transactions,
        FakeModel(),
    )

    assert len(scored) == 2
    assert "risk_score" in scored.columns
    assert "decision" in scored.columns
    assert "network_score" in scored.columns

    assert len(queue) >= 1
    assert queue.iloc[0]["decision"] in {"BLOCK", "REVIEW"}


def test_run_batch_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    transactions = pd.DataFrame(
        {
            "transaction_id": ["T1"],
            "account_id": ["A1"],
            "merchant_id": ["M1"],
            "device_id": ["D1"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 12:00:00"]
            ),
            "amount": [100.0],
            "ip_country": ["IN"],
            "shipping_country": ["IN"],
            "payment_method": ["upi"],
            "merchant_category": ["food"],
            "is_chargeback": [0],
        }
    )

    transaction_path = (
        tmp_path / "transactions.parquet"
    )

    transactions.to_parquet(
        transaction_path,
        index=False,
    )

    import razorguard.runner.batch as batch

    class FakeModel:
        def predict_proba(self, X):
            return pd.DataFrame(
                {
                    0: [0.9] * len(X),
                    1: [0.1] * len(X),
                }
            ).to_numpy()

    monkeypatch.setattr(
        batch,
        "load_model",
        lambda: (
            FakeModel(),
            {
                "model": "test_model",
                "threshold": 0.5,
            },
        ),
    )

    summary = run_batch(
        transactions_path=transaction_path,
        artifacts_path=tmp_path / "artifacts",
    )

    assert summary[
        "transactions"
    ] == 1

    assert (
        tmp_path
        / "artifacts"
        / "scored_transactions.parquet"
    ).exists()

    assert (
        tmp_path
        / "artifacts"
        / "investigator_cases.parquet"
    ).exists()

    summary_path = (
        tmp_path
        / "artifacts"
        / "run_summary.json"
    )

    assert summary_path.exists()

    loaded = json.loads(
        summary_path.read_text()
    )

    assert loaded[
        "scored_transactions"
    ] == 1