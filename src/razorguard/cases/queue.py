from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
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

VALID_PRIORITIES = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
}


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def _case_id(
    transaction_id: str,
) -> str:
    """Create a deterministic case ID."""

    if not transaction_id:
        raise ValueError(
            "transaction_id must not be empty"
        )

    return f"CASE-{transaction_id}"


def _priority_from_risk_level(
    risk_level: str,
) -> str:
    """
    Convert risk level into investigator priority.

    CaseStore uses the same four priority values:
        CRITICAL
        HIGH
        MEDIUM
        LOW
    """

    priority = str(
        risk_level
    ).upper()

    if priority not in VALID_PRIORITIES:
        return "MEDIUM"

    return priority


def _investigation_narrative(
    case: RiskCase,
) -> str:
    """
    Build a deterministic investigator-facing narrative.

    This is intentionally evidence-based and does not use an
    external AI service.
    """

    evidence = "; ".join(
        str(item)
        for item in case.evidence
        if str(item).strip()
    )

    return (
        f"Transaction {case.transaction_id} "
        f"received a {case.risk_level} risk classification "
        f"with a {case.decision} decision. "
        f"Primary reason: {case.primary_reason}. "
        f"Evidence: {evidence}"
    )


def cases_to_frame(
    cases: Iterable[RiskCase],
) -> pd.DataFrame:
    """
    Convert RiskCase objects into a complete investigator queue.

    The returned frame is intentionally compatible with the
    CaseStore schema so the persisted artifact can be consumed
    directly by the API and frontend.
    """

    rows = []

    for case in cases:
        now = _utc_now()

        row = asdict(case)

        risk_level = str(
            case.risk_level
        ).upper()

        if risk_level not in VALID_PRIORITIES:
            risk_level = "MEDIUM"

        evidence = [
            str(item)
            for item in case.evidence
        ]

        evidence_text = " | ".join(
            evidence
        )

        row = {
            # --------------------------------------------------
            # Investigator case identity
            # --------------------------------------------------
            "case_id": _case_id(
                case.transaction_id
            ),
            "transaction_id": str(
                case.transaction_id
            ),

            # --------------------------------------------------
            # Investigation lifecycle
            # --------------------------------------------------
            "status": "OPEN",
            "priority": _priority_from_risk_level(
                risk_level
            ),
            "assigned_to": None,
            "created_at": now,
            "updated_at": now,

            # --------------------------------------------------
            # Risk information
            # --------------------------------------------------
            "risk_score": float(
                case.risk_score
            ),
            "risk_level": risk_level,
            "decision": str(
                case.decision
            ),
            "primary_reason": str(
                case.primary_reason
            ),

            # --------------------------------------------------
            # Evidence
            # --------------------------------------------------
            "evidence": evidence,
            "evidence_text": evidence_text,

            # --------------------------------------------------
            # Model / network signals
            # --------------------------------------------------
            "model_probability": float(
                case.model_probability
            ),
            "network_score": float(
                case.network_score
            ),

            # --------------------------------------------------
            # Investigator narrative
            # --------------------------------------------------
            "investigation_narrative": (
                _investigation_narrative(case)
            ),
        }

        rows.append(row)

    columns = [
        "case_id",
        "transaction_id",
        "status",
        "priority",
        "assigned_to",
        "created_at",
        "updated_at",
        "risk_score",
        "risk_level",
        "decision",
        "primary_reason",
        "evidence",
        "evidence_text",
        "model_probability",
        "network_score",
        "investigation_narrative",
    ]

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


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
        5. higher risk severity first
        6. transaction ID for deterministic tie-breaking
    """

    frame = cases_to_frame(
        cases
    )

    if frame.empty:
        return frame

    if not include_allowed:
        frame = frame[
            frame["decision"].isin(
                [
                    "BLOCK",
                    "REVIEW",
                ]
            )
        ].copy()

    if frame.empty:
        return frame.reset_index(
            drop=True
        )

    frame["_decision_priority"] = (
        frame["decision"]
        .map(
            DECISION_PRIORITY
        )
        .fillna(99)
    )

    frame["_level_priority"] = (
        frame["risk_level"]
        .map(
            LEVEL_PRIORITY
        )
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
        .reset_index(
            drop=True
        )
    )

    frame.insert(
        0,
        "case_rank",
        range(
            1,
            len(frame) + 1,
        ),
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
        "cases": int(
            len(queue)
        ),
        "block_cases": int(
            (
                queue["decision"]
                == "BLOCK"
            ).sum()
        ),
        "review_cases": int(
            (
                queue["decision"]
                == "REVIEW"
            ).sum()
        ),
        "critical_cases": int(
            (
                queue["risk_level"]
                == "CRITICAL"
            ).sum()
        ),
        "high_cases": int(
            (
                queue["risk_level"]
                == "HIGH"
            ).sum()
        ),
        "medium_cases": int(
            (
                queue["risk_level"]
                == "MEDIUM"
            ).sum()
        ),
        "low_cases": int(
            (
                queue["risk_level"]
                == "LOW"
            ).sum()
        ),
        "mean_risk_score": round(
            float(
                queue[
                    "risk_score"
                ].mean()
            ),
            4,
        ),
    }


def write_case_queue(
    queue: pd.DataFrame,
    path: str,
) -> None:
    """
    Persist the investigator queue as a CaseStore-compatible
    Parquet artifact.

    The evidence list is removed before Parquet serialization
    because evidence_text is the portable representation.
    """

    output = queue.copy()

    # Remove the Python list representation.
    if "evidence" in output.columns:
        output = output.drop(
            columns=[
                "evidence"
            ]
        )

    # case_rank is useful for the ranked queue but is not part
    # of the canonical CaseStore schema. It can safely remain
    # in the Parquet artifact because CaseStore ignores unknown
    # columns.
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        path,
        index=False,
    )