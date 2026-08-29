from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from razorguard.api.models import (
    AuditEventResponse,
    AuditResponse,
    CaseAssignRequest,
    CaseListResponse,
    CaseResponse,
    CaseTransitionRequest,
    ErrorResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from razorguard.config import TRANSACTIONS_PATH
from razorguard.investigation.store import CaseStore
from razorguard.runner.batch import (
    load_model,
    score_runtime_transaction,
)
from razorguard.runtime.context import RuntimeContextStore


router = APIRouter(
    prefix="/v1",
    tags=["risk"],
)


DEFAULT_CASES_PATH = Path(
    "artifacts/investigator_cases.parquet"
)


# ============================================================
# RUNTIME HELPERS
# ============================================================


def _load_runtime_model():
    """Load the currently configured RazorGuard model."""

    try:
        return load_model()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


def _request_to_frame(
    request: TransactionScoreRequest,
) -> pd.DataFrame:
    """Convert a validated API request into a transaction frame."""

    return pd.DataFrame(
        [request.model_dump()]
    )


def _runtime_context_store() -> RuntimeContextStore:
    """Return the configured runtime transaction history provider."""

    return RuntimeContextStore(
        TRANSACTIONS_PATH
    )


def _case_store() -> CaseStore:
    """Return the configured investigator case store."""

    return CaseStore(
        DEFAULT_CASES_PATH
    )

def _get_existing_case_id(
    store: CaseStore,
    transaction_id: str,
) -> str | None:
    """Return an existing case ID for a transaction, if present."""

    case_id = f"CASE-{transaction_id}"

    case = store.get(
        case_id
    )

    if case is None:
        return None

    return str(
        case["case_id"]
    )


# ============================================================
# CASE RESPONSE
# ============================================================


def _case_response(
    case: dict[str, Any],
) -> CaseResponse:
    """Convert a CaseStore record into an API response."""

    return CaseResponse(
        case_id=str(
            case["case_id"]
        ),
        transaction_id=str(
            case["transaction_id"]
        ),
        status=str(
            case["status"]
        ),
        priority=str(
            case["priority"]
        ),
        assigned_to=(
            None
            if case.get("assigned_to") is None
            else str(
                case["assigned_to"]
            )
        ),
        created_at=str(
            case["created_at"]
        ),
        updated_at=str(
            case["updated_at"]
        ),
        risk_score=float(
            case["risk_score"]
        ),
        risk_level=str(
            case["risk_level"]
        ),
        decision=str(
            case["decision"]
        ),
        primary_reason=str(
            case["primary_reason"]
        ),
        evidence_text=str(
            case["evidence_text"]
        ),
        model_probability=float(
            case["model_probability"]
        ),
        network_score=float(
            case["network_score"]
        ),
        investigation_narrative=str(
            case["investigation_narrative"]
        ),
    )


# ============================================================
# TRANSACTION SCORING
# ============================================================


@router.post(
    "/transactions/score",
    response_model=TransactionScoreResponse,
    responses={
        503: {
            "model": ErrorResponse,
        },
        422: {
            "description": "Invalid transaction payload",
        },
    },
)
def score_transaction_api(
    request: TransactionScoreRequest,
) -> TransactionScoreResponse:
    """
    Score one live transaction using runtime historical context.

    Historical transactions strictly before the incoming
    transaction are supplied to the point-in-time feature engine.

    The incoming transaction itself is scored as the final event.

    Non-ALLOW decisions automatically create an investigator case.
    """

    model, metadata = _load_runtime_model()

    raw = _request_to_frame(
        request
    )

    transaction = (
        raw.iloc[0]
        .to_dict()
    )

    # --------------------------------------------------------
    # Runtime scoring
    # --------------------------------------------------------

    try:
        context_store = (
            _runtime_context_store()
        )

        result_data = (
            score_runtime_transaction(
                transaction=transaction,
                model=model,
                context_store=context_store,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "transaction scoring failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    if not result_data:
        raise HTTPException(
            status_code=500,
            detail=(
                "transaction scoring "
                "produced no result"
            ),
        )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_text = str(
        result_data.get(
            "evidence_text",
            "",
        )
    )

    evidence = [
        item.strip()
        for item in evidence_text.split("|")
        if item.strip()
    ]

        # --------------------------------------------------------
    # Automatic investigator case creation
    # --------------------------------------------------------

    case_id: str | None = None

    if result_data["decision"] != "ALLOW":
        try:
            store = _case_store()

            transaction_id = str(
                result_data["transaction_id"]
            )

            existing_case_id = (
                _get_existing_case_id(
                    store,
                    transaction_id,
                )
            )

            if existing_case_id is not None:
                # Idempotent retry:
                # the investigation case already exists.
                case_id = existing_case_id

            else:
                created_case = store.create(
                    result_data["case"],
                    actor="api",
                    investigation_narrative=(
                        result_data["case"].get(
                            "investigation_narrative",
                            "",
                        )
                    ),
                )

                case_id = str(
                    created_case["case_id"]
                )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "case creation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return TransactionScoreResponse(
        transaction_id=str(
            result_data[
                "transaction_id"
            ]
        ),
        risk_score=float(
            result_data[
                "risk_score"
            ]
        ),
        risk_level=str(
            result_data[
                "risk_level"
            ]
        ),
        decision=str(
            result_data[
                "decision"
            ]
        ),
        primary_reason=str(
            result_data[
                "primary_reason"
            ]
        ),
        evidence=evidence,
        model_probability=float(
            result_data[
                "model_probability"
            ]
        ),
        network_score=float(
            result_data[
                "network_score"
            ]
        ),
        behavioral_signal=float(
            result_data[
                "behavioral_signal"
            ]
        ),
        model=str(
            metadata.get(
                "model",
                "unknown",
            )
        ),
        model_threshold=float(
            metadata.get(
                "threshold",
                0.0,
            )
        ),
        case_id=case_id,
    )


# ============================================================
# CASE MANAGEMENT
# ============================================================


@router.get(
    "/cases",
    response_model=CaseListResponse,
)
def list_cases(
    status: str | None = Query(
        default=None
    ),
    assigned_to: str | None = Query(
        default=None
    ),
    priority: str | None = Query(
        default=None
    ),
) -> CaseListResponse:
    """
    List investigator cases with optional filters.
    """

    store = _case_store()

    try:
        frame = store.list(
            status=status,
            assigned_to=assigned_to,
            priority=priority,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    cases = [
        _case_response(
            row.to_dict()
        )
        for _, row in frame.iterrows()
    ]

    return CaseListResponse(
        cases=cases,
        total=len(cases),
    )


@router.get(
    "/cases/{case_id}",
    response_model=CaseResponse,
)
def get_case(
    case_id: str,
) -> CaseResponse:
    """
    Retrieve one investigator case.
    """

    store = _case_store()

    case = store.get(
        case_id
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"case not found: "
                f"{case_id}"
            ),
        )

    return _case_response(
        case
    )


@router.post(
    "/cases/{case_id}/assign",
    response_model=CaseResponse,
)
def assign_case_api(
    case_id: str,
    request: CaseAssignRequest,
) -> CaseResponse:
    """
    Assign an investigator case.
    """

    store = _case_store()

    try:
        case = store.assign(
            case_id,
            request.investigator,
            actor=request.actor,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _case_response(
        case
    )


@router.post(
    "/cases/{case_id}/transition",
    response_model=CaseResponse,
)
def transition_case_api(
    case_id: str,
    request: CaseTransitionRequest,
) -> CaseResponse:
    """
    Transition a case through the investigation lifecycle.
    """

    store = _case_store()

    try:
        case = store.update_status(
            case_id,
            request.status,
            actor=request.actor,
            details=request.details,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _case_response(
        case
    )


# ============================================================
# AUDIT
# ============================================================


@router.get(
    "/cases/{case_id}/audit",
    response_model=AuditResponse,
)
def get_case_audit_api(
    case_id: str,
) -> AuditResponse:
    """
    Retrieve the append-only audit history for a case.
    """

    store = _case_store()

    if store.get(case_id) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"case not found: "
                f"{case_id}"
            ),
        )

    frame = store.audit(
        case_id
    )

    events = [
        AuditEventResponse(
            case_id=str(
                row["case_id"]
            ),
            timestamp=str(
                row["timestamp"]
            ),
            action=str(
                row["action"]
            ),
            actor=str(
                row["actor"]
            ),
            from_status=(
                None
                if pd.isna(
                    row["from_status"]
                )
                else str(
                    row["from_status"]
                )
            ),
            to_status=(
                None
                if pd.isna(
                    row["to_status"]
                )
                else str(
                    row["to_status"]
                )
            ),
            details=str(
                row["details"]
            ),
        )
        for _, row in frame.iterrows()
    ]

    return AuditResponse(
        case_id=case_id,
        events=events,
        total=len(events),
    )