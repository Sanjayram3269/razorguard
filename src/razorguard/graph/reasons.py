from __future__ import annotations

import pandas as pd


def network_risk_reasons(row: pd.Series) -> list[str]:
    """
    Convert network features into concise investigation evidence.
    """

    reasons: list[str] = []

    shared_accounts = int(
        row.get("prior_accounts_per_device", 0)
    )

    shared_merchants = int(
        row.get("prior_accounts_per_merchant", 0)
    )

    device_novelty = int(
        row.get("account_device_novelty", 0)
    )

    merchant_novelty = int(
        row.get("account_merchant_novelty", 0)
    )

    if shared_accounts >= 3:
        reasons.append(
            f"device previously associated with "
            f"{shared_accounts} accounts"
        )

    elif shared_accounts >= 2:
        reasons.append(
            "device shared across multiple accounts"
        )

    if device_novelty:
        reasons.append(
            "device is new for this account"
        )

    if (
        shared_accounts >= 2
        and device_novelty
    ):
        reasons.append(
            "new device belongs to an existing shared-device network"
        )

    if shared_merchants >= 10 and merchant_novelty:
        reasons.append(
            f"new merchant relationship with an entity "
            f"used by {shared_merchants} prior accounts"
        )

    if (
        row.get(
            "shared_merchant_novelty_risk",
            0,
        )
        > 0
    ):
        reasons.append(
            "merchant relationship overlaps with other accounts"
        )

    if (
        row.get("network_risk_score", 0)
        >= 2.0
    ):
        reasons.append(
            "elevated network-level risk"
        )

    return reasons