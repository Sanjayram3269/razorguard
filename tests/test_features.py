import pandas as pd

from razorguard.ml.features import add_historical_features


def test_historical_count_never_uses_current_event():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "account_id": ["a", "a", "a"],
            "merchant_id": ["m", "m", "m"],
            "device_id": ["d", "d", "d"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-02 12:00:00",
                    "2026-01-03 12:00:00",
                ]
            ),
            "amount": [100.0, 200.0, 300.0],
            "ip_country": ["IN"] * 3,
            "shipping_country": ["IN"] * 3,
        }
    )

    out = add_historical_features(df)

    # Every transaction has two previous transactions for the
    # account, regardless of the velocity window.
    assert out["prior_tx_count"].tolist() == [0, 1, 2]

    # Transactions are exactly 24 hours apart, so none of the
    # previous events should appear inside a 60-minute window.
    assert out[
        "device_id_prior_count_60m"
    ].tolist() == [0, 0, 0]

    assert out[
        "merchant_id_prior_count_60m"
    ].tolist() == [0, 0, 0]

    # Historical entity count DOES retain long-term history.
    assert out[
        "prior_accounts_per_device"
    ].tolist() == [0, 1, 1]


def test_current_transaction_is_not_in_velocity_features():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "account_id": ["a", "a"],
            "merchant_id": ["m", "m"],
            "device_id": ["d", "d"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:02:00",
                ]
            ),
            "amount": [100.0, 900.0],
            "ip_country": ["IN", "IN"],
            "shipping_country": ["IN", "IN"],
        }
    )

    out = add_historical_features(df)

    # First event sees no history.
    assert out.loc[0, "account_id_prior_count_5m"] == 0
    assert out.loc[0, "account_id_prior_amount_5m"] == 0

    # Second event sees ONLY the first event.
    assert out.loc[1, "account_id_prior_count_5m"] == 1
    assert out.loc[1, "account_id_prior_amount_5m"] == 100

    # If the current $900 transaction had leaked into the window,
    # the amount would incorrectly be 1000.
    assert out.loc[1, "account_id_prior_amount_5m"] != 1000


def test_velocity_excludes_events_outside_window():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "account_id": ["a", "a", "a"],
            "merchant_id": ["m", "m", "m"],
            "device_id": ["d", "d", "d"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:30:00",
                    "2026-01-01 13:01:00",
                ]
            ),
            "amount": [100.0, 200.0, 300.0],
            "ip_country": ["IN"] * 3,
            "shipping_country": ["IN"] * 3,
        }
    )

    out = add_historical_features(df)

    # t2 is 30 minutes after t1 -> t1 belongs to the 60m window.
    assert out.loc[1, "account_id_prior_count_60m"] == 1

    # t3 is 31 minutes after t2, but 61 minutes after t1.
    # Therefore only t2 remains in the 60m window.
    assert out.loc[2, "account_id_prior_count_60m"] == 1
    assert out.loc[2, "account_id_prior_amount_60m"] == 200


def test_chargeback_label_is_not_a_runtime_feature():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1"],
            "account_id": ["a"],
            "merchant_id": ["m"],
            "device_id": ["d"],
            "timestamp": pd.to_datetime(["2026-01-01"]),
            "amount": [100.0],
            "ip_country": ["IN"],
            "shipping_country": ["IN"],
            "is_chargeback": [1],
        }
    )

    out = add_historical_features(df)

    assert "is_chargeback" in out.columns

    feature_columns = [
        column
        for column in out.columns
        if column not in {
            "is_chargeback",
            "transaction_id",
            "timestamp",
            "previous_timestamp",
        }
    ]

    assert "is_chargeback" not in feature_columns

def test_behavioral_features_are_point_in_time():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "account_id": ["a", "a", "a"],
            "merchant_id": ["m1", "m2", "m3"],
            "device_id": ["d", "d", "d"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:01:00",
                    "2026-01-01 12:02:00",
                ]
            ),
            "amount": [100.0, 100.0, 100.0],
            "ip_country": ["IN"] * 3,
            "shipping_country": ["IN"] * 3,
        }
    )

    out = add_historical_features(df)

    assert out["prior_unique_merchants"].tolist() == [
        0,
        1,
        2,
    ]

    assert out[
        "account_id_prior_unique_merchant_id_60m"
    ].tolist() == [
        0,
        1,
        2,
    ]


def test_dormancy_feature_detects_long_gap():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "account_id": ["a", "a"],
            "merchant_id": ["m", "m"],
            "device_id": ["d", "d"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-10 12:00:00",
                ]
            ),
            "amount": [100.0, 300.0],
            "ip_country": ["IN", "IN"],
            "shipping_country": ["IN", "IN"],
        }
    )

    out = add_historical_features(df)

    assert out.loc[0, "is_dormant_return"] == 0
    assert out.loc[1, "is_dormant_return"] == 1
    assert out.loc[1, "is_long_dormancy"] == 0