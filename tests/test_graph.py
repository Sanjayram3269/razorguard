import pandas as pd

from razorguard.graph.features import add_graph_features


def test_graph_features_are_point_in_time():

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
                "a1",
            ],
            "device_id": [
                "d1",
                "d1",
                "d1",
            ],
            "merchant_id": [
                "m1",
                "m2",
                "m1",
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

    out = add_graph_features(df)

    # t1 sees nothing.
    assert out.loc[
        0,
        "prior_accounts_per_device",
    ] == 0

    # t2 sees a1 on d1.
    assert out.loc[
        1,
        "prior_accounts_per_device",
    ] == 1

    # t3 sees both a1 and a2 on d1.
    assert out.loc[
        2,
        "prior_accounts_per_device",
    ] == 2


def test_graph_features_do_not_use_current_event():

    df = pd.DataFrame(
        {
            "transaction_id": [
                "t1",
                "t2",
            ],
            "account_id": [
                "a1",
                "a1",
            ],
            "device_id": [
                "d1",
                "d2",
            ],
            "merchant_id": [
                "m1",
                "m2",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
        }
    )

    out = add_graph_features(df)

    assert out.loc[
        0,
        "prior_devices_per_account",
    ] == 0

    assert out.loc[
        1,
        "prior_devices_per_account",
    ] == 1

    assert out.loc[
        1,
        "account_device_novelty",
    ] == 1