from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def add_graph_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add point-in-time graph relationship features.

    Every feature only uses transactions observed BEFORE
    the current transaction.
    """

    df = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    account_devices = defaultdict(set)
    account_merchants = defaultdict(set)

    device_accounts = defaultdict(set)
    device_merchants = defaultdict(set)

    merchant_accounts = defaultdict(set)
    merchant_devices = defaultdict(set)

    features = {
        "prior_devices_per_account": [],
        "prior_merchants_per_account": [],
        "prior_accounts_per_device": [],
        "prior_merchants_per_device": [],
        "prior_accounts_per_merchant": [],
        "prior_devices_per_merchant": [],
        "device_shared_with_accounts": [],
        "merchant_shared_with_accounts": [],
        "account_device_novelty": [],
        "account_merchant_novelty": [],
    }

    for row in df.itertuples(index=False):

        account = row.account_id
        device = row.device_id
        merchant = row.merchant_id

        features[
            "prior_devices_per_account"
        ].append(
            len(account_devices[account])
        )

        features[
            "prior_merchants_per_account"
        ].append(
            len(account_merchants[account])
        )

        features[
            "prior_accounts_per_device"
        ].append(
            len(device_accounts[device])
        )

        features[
            "prior_merchants_per_device"
        ].append(
            len(device_merchants[device])
        )

        features[
            "prior_accounts_per_merchant"
        ].append(
            len(merchant_accounts[merchant])
        )

        features[
            "prior_devices_per_merchant"
        ].append(
            len(merchant_devices[merchant])
        )

        features[
            "device_shared_with_accounts"
        ].append(
            max(
                0,
                len(device_accounts[device]) - 1,
            )
        )

        features[
            "merchant_shared_with_accounts"
        ].append(
            max(
                0,
                len(merchant_accounts[merchant]) - 1,
            )
        )

        features[
            "account_device_novelty"
        ].append(
            int(
                device
                not in account_devices[account]
            )
        )

        features[
            "account_merchant_novelty"
        ].append(
            int(
                merchant
                not in account_merchants[account]
            )
        )

        account_devices[account].add(device)
        account_merchants[account].add(merchant)

        device_accounts[device].add(account)
        device_merchants[device].add(merchant)

        merchant_accounts[merchant].add(account)
        merchant_devices[merchant].add(device)

    for name, values in features.items():
        df[name] = np.asarray(
            values,
            dtype=np.int32,
        )

    return df


GRAPH_FEATURES = [
    "prior_devices_per_account",
    "prior_merchants_per_account",
    "prior_accounts_per_device",
    "prior_merchants_per_device",
    "prior_accounts_per_merchant",
    "prior_devices_per_merchant",
    "device_shared_with_accounts",
    "merchant_shared_with_accounts",
    "account_device_novelty",
    "account_merchant_novelty",
]