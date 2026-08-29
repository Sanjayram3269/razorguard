from __future__ import annotations

from collections import defaultdict

import pandas as pd


NODE_COLUMNS = {
    "account": "account_id",
    "device": "device_id",
    "merchant": "merchant_id",
}


def build_entity_graph(
    transactions: pd.DataFrame,
) -> dict:
    """
    Build a lightweight temporal entity graph.

    Nodes:
        account
        device
        merchant

    Edges:
        account -> device
        account -> merchant
        device -> merchant

    The graph is constructed from observed transactions and is
    intended for investigation-time relationship features.
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
        transactions[
            list(required)
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    nodes = {
        "account": set(),
        "device": set(),
        "merchant": set(),
    }

    edges = {
        "account_device": defaultdict(set),
        "account_merchant": defaultdict(set),
        "device_account": defaultdict(set),
        "device_merchant": defaultdict(set),
        "merchant_account": defaultdict(set),
        "merchant_device": defaultdict(set),
    }

    for row in df.itertuples(index=False):

        account = row.account_id
        device = row.device_id
        merchant = row.merchant_id

        nodes["account"].add(account)
        nodes["device"].add(device)
        nodes["merchant"].add(merchant)

        edges["account_device"][account].add(device)
        edges["device_account"][device].add(account)

        edges["account_merchant"][account].add(merchant)
        edges["merchant_account"][merchant].add(account)

        edges["device_merchant"][device].add(merchant)
        edges["merchant_device"][merchant].add(device)

    return {
        "nodes": nodes,
        "edges": edges,
    }


def graph_summary(graph: dict) -> dict:
    """
    Return deterministic graph-level statistics.
    """

    nodes = graph["nodes"]
    edges = graph["edges"]

    return {
        "accounts": len(nodes["account"]),
        "devices": len(nodes["device"]),
        "merchants": len(nodes["merchant"]),
        "account_device_edges": sum(
            len(v)
            for v in edges["account_device"].values()
        ),
        "account_merchant_edges": sum(
            len(v)
            for v in edges["account_merchant"].values()
        ),
        "device_merchant_edges": sum(
            len(v)
            for v in edges["device_merchant"].values()
        ),
    }