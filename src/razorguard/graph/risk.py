from __future__ import annotations

import numpy as np
import pandas as pd

from razorguard.graph.features import add_graph_features


def add_network_risk_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add point-in-time network-risk features.

    The scoring deliberately separates ordinary entity popularity
    from suspicious relationship patterns.
    """

    df = add_graph_features(transactions)

    # ------------------------------------------------------------------
    # Raw relationship strength
    # ------------------------------------------------------------------

    df["device_network_risk"] = np.log1p(
        df["prior_accounts_per_device"]
    )

    df["merchant_network_exposure"] = np.log1p(
        df["prior_accounts_per_merchant"]
    )

    df["device_merchant_network_risk"] = np.log1p(
        df["prior_merchants_per_device"]
    )

    # ------------------------------------------------------------------
    # Suspicious relationship patterns
    # ------------------------------------------------------------------

    df["shared_device_novelty_risk"] = (
        df["device_shared_with_accounts"]
        * df["account_device_novelty"]
    )

    df["shared_merchant_novelty_risk"] = (
        df["merchant_shared_with_accounts"]
        * df["account_merchant_novelty"]
    )

    # ------------------------------------------------------------------
    # Entity popularity correction
    #
    # A merchant used by hundreds of accounts is not automatically
    # suspicious. Popularity should contribute weakly.
    # ------------------------------------------------------------------

    merchant_popularity = (
        df["prior_accounts_per_merchant"]
        .clip(lower=0)
    )

    device_shared_accounts = (
        df["prior_accounts_per_device"]
        .clip(lower=0)
    )

    # Saturating transformations prevent very large entities from
    # dominating the entire risk score.
    df["merchant_popularity_adjusted"] = (
        np.sqrt(merchant_popularity)
    )

    df["device_sharing_adjusted"] = (
        np.sqrt(device_shared_accounts)
    )

    # ------------------------------------------------------------------
    # Composite network score
    # ------------------------------------------------------------------

    df["network_risk_score"] = (
        0.20 * df["device_network_risk"]
        + 0.08 * df["merchant_popularity_adjusted"]
        + 0.18 * df["device_merchant_network_risk"]
        + 0.30 * df["shared_device_novelty_risk"]
        + 0.24 * df["shared_merchant_novelty_risk"]
    )

    return df


NETWORK_RISK_FEATURES = [
    "device_network_risk",
    "merchant_popularity_adjusted",
    "device_merchant_network_risk",
    "shared_device_novelty_risk",
    "shared_merchant_novelty_risk",
    "network_risk_score",
]