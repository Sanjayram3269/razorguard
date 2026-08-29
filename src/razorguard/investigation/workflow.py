from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from razorguard.cases.enrichment import enrich_case
from razorguard.cases.queue import rank_case_queue
from razorguard.engine.pipeline import behavioral_signal, score_transaction
from razorguard.graph.risk import add_network_risk_features
from razorguard.investigation.store import CaseStore
from razorguard.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_frame,
)


def create_investigation_cases(
    transactions: pd.DataFrame,
    model: Any,
    store: CaseStore,
    *,
    actor: str = "system",
    include_allowed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run risk scoring and persist investigator cases.

    Pipeline:

        transactions
            -> model features
            -> ML probability
            -> network risk
            -> behavioral signal
            -> deterministic risk fusion
            -> case enrichment
            -> persistent CaseStore
            -> investigator queue

    Returns:
        scored transaction frame
        investigator queue
    """

    if transactions.empty:
        empty_scored = pd.DataFrame(
            columns=[
                "transaction_id",
                "model_probability",
                "network_score",
                "behavioral_signal",
                "risk_score",
                "risk_level",
                "decision",
                "primary_reason",
                "evidence_text",
            ]
        )

        return empty_scored, rank_case_queue(
            [],
            include_allowed=False,
        )

    model_frame = build_model_frame(
        transactions
    )

    network_frame = add_network_risk_features(
        transactions
    )

    model_input = model_frame[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    probabilities = model.predict_proba(
        model_input
    )[:, 1]

    network_scores = (
        network_frame["network_risk_score"]
        .to_numpy(dtype=float)
    )

    scored_rows: list[dict[str, Any]] = []
    persisted_cases = []

    for index, row in model_frame.iterrows():
        transaction = row.to_dict()

        probability = float(
            probabilities[index]
        )

        network_score = float(
            network_scores[index]
        )

        behavior = behavioral_signal(
            row
        )

        case = score_transaction(
            transaction=transaction,
            model_probability=probability,
            network_score=network_score,
            behavioral_signal=behavior,
        )

        enriched = enrich_case(
            case=case,
            transaction=transaction,
            behavioral_signal=behavior,
        )

        # ALLOW decisions are excluded from the persisted
        # investigator cases unless explicitly requested.
        if include_allowed or case.decision != "ALLOW":
            store.create(
                enriched,
                actor=actor,
                investigation_narrative=enriched[
                    "investigation_narrative"
                ],
            )

            persisted_cases.append(
                case
            )

        scored_rows.append(
            {
                "transaction_id": case.transaction_id,
                "model_probability": case.model_probability,
                "network_score": case.network_score,
                "behavioral_signal": round(
                    behavior,
                    6,
                ),
                "risk_score": case.risk_score,
                "risk_level": case.risk_level,
                "decision": case.decision,
                "primary_reason": case.primary_reason,
                "evidence_text": " | ".join(
                    case.evidence
                ),
            }
        )

    scored = pd.DataFrame(
        scored_rows
    )

    # Investigator queue contains only actionable
    # BLOCK/REVIEW cases. ALLOW cases may be persisted
    # for audit/history but never enter this queue.
    queue = rank_case_queue(
        persisted_cases,
        include_allowed=False,
    )

    return scored, queue


def run_investigation_workflow(
    transactions_path: str | Path,
    model: Any,
    cases_path: str | Path,
    *,
    actor: str = "system",
    include_allowed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete investigation workflow from a transaction file.

    The CaseStore remains the source of truth for investigator state.
    """

    transactions_path = Path(
        transactions_path
    )

    store = CaseStore(
        cases_path
    )

    transactions = pd.read_parquet(
        transactions_path
    )

    return create_investigation_cases(
        transactions=transactions,
        model=model,
        store=store,
        actor=actor,
        include_allowed=include_allowed,
    )


def reopen_case(
    cases_path: str | Path,
    case_id: str,
) -> dict[str, Any] | None:
    """Reopen a persisted case record for investigation."""

    store = CaseStore(
        cases_path
    )

    return store.get(
        case_id
    )


def assign_case(
    cases_path: str | Path,
    case_id: str,
    investigator: str,
    *,
    actor: str = "lead",
) -> dict[str, Any]:
    """Assign a persisted case to an investigator."""

    store = CaseStore(
        cases_path
    )

    return store.assign(
        case_id,
        investigator,
        actor=actor,
    )


def transition_case(
    cases_path: str | Path,
    case_id: str,
    status: str,
    *,
    actor: str = "investigator",
    details: str = "",
) -> dict[str, Any]:
    """Transition a persisted case through the investigation lifecycle."""

    store = CaseStore(
        cases_path
    )

    return store.update_status(
        case_id,
        status,
        actor=actor,
        details=details,
    )


def get_case_audit(
    cases_path: str | Path,
    case_id: str,
) -> pd.DataFrame:
    """Retrieve the complete audit history for a persisted case."""

    store = CaseStore(
        cases_path
    )

    return store.audit(
        case_id
    )