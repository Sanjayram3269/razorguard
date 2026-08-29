from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from razorguard.api.models import (
    ErrorResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from razorguard.runner.batch import load_model, score_batch


router = APIRouter(
    prefix="/v1",
    tags=["risk"],
)


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
    Score one transaction through the existing RazorGuard engine.

    The API deliberately delegates to the already-tested batch
    scoring path so that API and offline scoring use identical
    feature engineering, network risk, behavioral scoring,
    risk fusion, evidence, and policy logic.
    """

    model, metadata = _load_runtime_model()

    raw = _request_to_frame(
        request
    )

    try:
        scored, _ = score_batch(
            raw,
            model,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "transaction scoring failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    if scored.empty:
        raise HTTPException(
            status_code=500,
            detail="transaction scoring produced no result",
        )

    result = scored.iloc[0]

    evidence_text = result.get(
        "evidence_text",
        "",
    )

    evidence = [
        item.strip()
        for item in str(
            evidence_text
        ).split("|")
        if item.strip()
    ]

    return TransactionScoreResponse(
        transaction_id=str(
            result["transaction_id"]
        ),
        risk_score=float(
            result["risk_score"]
        ),
        risk_level=str(
            result["risk_level"]
        ),
        decision=str(
            result["decision"]
        ),
        primary_reason=str(
            result["primary_reason"]
        ),
        evidence=evidence,
        model_probability=float(
            result["model_probability"]
        ),
        network_score=float(
            result["network_score"]
        ),
        behavioral_signal=float(
            result["behavioral_signal"]
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
    )