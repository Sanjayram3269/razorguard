from __future__ import annotations

from collections import defaultdict

import pandas as pd


def investigate_transaction(
    transactions: pd.DataFrame,
    transaction_id: str,
) -> dict:
    """
    Produce an explainable investigation context for one transaction.

    The investigation describes relationships around the account,
    device and merchant involved in the event.
    """

    required = {
        "transaction_id",
        "account_id",
        "device_id",
        "merchant_id",
        "timestamp",
    }

    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    matches = df.index[
        df["transaction_id"]
        == transaction_id
    ].tolist()

    if not matches:
        raise KeyError(
            f"Unknown transaction_id: {transaction_id}"
        )

    index = matches[0]

    current = df.iloc[index]

    history = df.iloc[:index]

    account_id = current["account_id"]
    device_id = current["device_id"]
    merchant_id = current["merchant_id"]

    account_history = history[
        history["account_id"]
        == account_id
    ]

    device_accounts = (
        history.loc[
            history["device_id"]
            == device_id,
            "account_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    merchant_accounts = (
        history.loc[
            history["merchant_id"]
            == merchant_id,
            "account_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    related_transactions = history[
        (
            history["device_id"]
            == device_id
        )
        |
        (
            history["merchant_id"]
            == merchant_id
        )
    ]

    return {
        "transaction_id": transaction_id,
        "timestamp": str(
            current["timestamp"]
        ),
        "account_id": account_id,
        "device_id": device_id,
        "merchant_id": merchant_id,
        "account_history_count": int(
            len(account_history)
        ),
        "accounts_seen_on_device": device_accounts,
        "accounts_seen_at_merchant": merchant_accounts,
        "related_transaction_count": int(
            len(related_transactions)
        ),
        "network_risk_signals": {
            "device_shared": len(
                device_accounts
            ) >= 2,
            "merchant_shared": len(
                merchant_accounts
            ) >= 3,
            "new_device_for_account": not (
                account_history["device_id"]
                == device_id
            ).any(),
            "new_merchant_for_account": not (
                account_history["merchant_id"]
                == merchant_id
            ).any(),
        },
    }