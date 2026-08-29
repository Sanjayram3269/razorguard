from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from razorguard.risk.case import RiskCase


DECISION_PRIORITY = {
    "BLOCK": 0,
    "REVIEW": 1,
    "ALLOW": 2,
}

LEVEL_PRIORITY = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


def cases_to_frame(
    cases: Iterable[RiskCase],
) -> pd.DataFrame:
    """
    Convert RiskCase objects into a tabular investigator queue.
    """

    rows = []

    for case in cases:
        row = asdict(case)

        # Keep evidence machine-readable while also exposing
        # a convenient human-readable representation.
        row["evidence"] = list(case.evidence)
        row["evidence_text"] = " | ".join(case.evidence)

        rows.append(row)

    columns = [
        "transaction_id",
        "risk_score",
        "risk_level",
        "decision",
        "primary_reason",
        "evidence",
        "evidence_text",
        "model_probability",
        "network_score",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def rank_case_queue(
    cases: Iterable[RiskCase],
    include_allowed: bool = False,
) -> pd.DataFrame:
    """
    Produce a deterministic investigator queue.

    Highest-risk actionable cases appear first.

    Ordering:
        1. BLOCK before REVIEW before ALLOW
        2. higher risk score first
        3. higher model probability first
        4. higher network score first
        5. transaction_id for deterministic tie-breaking
    """

    frame = cases_to_frame(cases)

    if frame.empty:
        return frame

    if not include_allowed:
        frame = frame[
            frame["decision"].isin(
                ["BLOCK", "REVIEW"]
            )
        ].copy()

    if frame.empty:
        return frame.reset_index(drop=True)

    frame["_decision_priority"] = (
        frame["decision"]
        .map(DECISION_PRIORITY)
        .fillna(99)
    )

    frame["_level_priority"] = (
        frame["risk_level"]
        .map(LEVEL_PRIORITY)
        .fillna(99)
    )

    frame = (
        frame
        .sort_values(
            [
                "_decision_priority",
                "risk_score",
                "model_probability",
                "network_score",
                "_level_priority",
                "transaction_id",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_decision_priority",
                "_level_priority",
            ]
        )
        .reset_index(drop=True)
    )

    frame.insert(
        0,
        "case_rank",
        range(1, len(frame) + 1),
    )

    return frame


def summarize_case_queue(
    queue: pd.DataFrame,
) -> dict[str, int | float]:
    """
    Return compact operational metrics for an investigator queue.
    """

    if queue.empty:
        return {
            "cases": 0,
            "block_cases": 0,
            "review_cases": 0,
            "critical_cases": 0,
            "high_cases": 0,
            "medium_cases": 0,
            "low_cases": 0,
            "mean_risk_score": 0.0,
        }

    return {
        "cases": int(len(queue)),
        "block_cases": int(
            (queue["decision"] == "BLOCK").sum()
        ),
        "review_cases": int(
            (queue["decision"] == "REVIEW").sum()
        ),
        "critical_cases": int(
            (queue["risk_level"] == "CRITICAL").sum()
        ),
        "high_cases": int(
            (queue["risk_level"] == "HIGH").sum()
        ),
        "medium_cases": int(
            (queue["risk_level"] == "MEDIUM").sum()
        ),
        "low_cases": int(
            (queue["risk_level"] == "LOW").sum()
        ),
        "mean_risk_score": round(
            float(queue["risk_score"].mean()),
            4,
        ),
    }


def write_case_queue(
    queue: pd.DataFrame,
    path: str,
) -> None:
    """
    Persist investigator queue as Parquet.
    """

    output = queue.copy()

    # Parquet does not need Python object lists for the primary
    # investigator artifact. Keep evidence_text as the portable
    # representation.
    if "evidence" in output.columns:
        output = output.drop(columns=["evidence"])

    output.to_parquet(
        path,
        index=False,
    )