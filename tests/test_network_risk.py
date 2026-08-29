import pandas as pd

from razorguard.graph.reasons import network_risk_reasons
from razorguard.graph.risk import add_network_risk_features


def test_network_risk_is_point_in_time():

    df = pd.DataFrame(
        {
            "transaction_id": [
                "t1",
                "t2",
                "t3",
            ],
            "account_id": [
                "a1",
                "a2",
                "a3",
            ],
            "device_id": [
                "d1",
                "d1",
                "d1",
            ],
            "merchant_id": [
                "m1",
                "m1",
                "m2",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 12:00:00",
                ]
            ),
        }
    )

    out = add_network_risk_features(df)

    assert out.loc[
        0,
        "prior_accounts_per_device",
    ] == 0

    assert out.loc[
        1,
        "prior_accounts_per_device",
    ] == 1

    assert out.loc[
        2,
        "prior_accounts_per_device",
    ] == 2


def test_network_score_is_non_negative():

    df = pd.DataFrame(
        {
            "transaction_id": ["t1"],
            "account_id": ["a1"],
            "device_id": ["d1"],
            "merchant_id": ["m1"],
            "timestamp": pd.to_datetime(
                ["2026-01-01"]
            ),
        }
    )

    out = add_network_risk_features(df)

    assert out.loc[
        0,
        "network_risk_score",
    ] >= 0


def test_network_reasons_are_explainable():

    row = pd.Series(
        {
            "prior_accounts_per_device": 4,
            "prior_accounts_per_merchant": 15,
            "account_device_novelty": 1,
            "account_merchant_novelty": 1,
            "shared_device_novelty_risk": 4,
            "network_risk_score": 3.2,
        }
    )

    reasons = network_risk_reasons(row)

    assert len(reasons) >= 3
    assert any(
        "device" in reason
        for reason in reasons
    )

def test_popular_merchant_does_not_dominate_network_score():

    df = pd.DataFrame(
        {
            "transaction_id": [
                f"t{i}"
                for i in range(20)
            ],
            "account_id": [
                f"a{i}"
                for i in range(20)
            ],
            "device_id": [
                f"d{i}"
                for i in range(20)
            ],
            "merchant_id": [
                "popular"
                for _ in range(20)
            ],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=20,
                freq="h",
            ),
        }
    )

    out = add_network_risk_features(df)

    # A highly popular merchant alone should not create
    # an extreme network risk score.
    assert (
        out["network_risk_score"].max()
        < 5
    )