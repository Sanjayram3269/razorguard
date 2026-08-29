from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TransactionScoreRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)

    timestamp: datetime

    amount: float = Field(ge=0)

    ip_country: str
    shipping_country: str
    payment_method: str
    merchant_category: str


class TransactionScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    decision: str
    primary_reason: str
    evidence: list[str]

    model_probability: float
    network_score: float
    behavioral_signal: float

    model: str
    model_threshold: float

    case_id: str | None = None


class CaseResponse(BaseModel):
    case_id: str
    transaction_id: str
    status: str
    priority: str
    assigned_to: str | None = None

    created_at: str
    updated_at: str

    risk_score: float
    risk_level: str
    decision: str
    primary_reason: str
    evidence_text: str

    model_probability: float
    network_score: float
    investigation_narrative: str


class CaseListResponse(BaseModel):
    cases: list[CaseResponse]
    total: int


class CaseAssignRequest(BaseModel):
    investigator: str = Field(min_length=1, max_length=200)
    actor: str = Field(default="system", min_length=1, max_length=200)


class CaseTransitionRequest(BaseModel):
    status: str = Field(min_length=1)
    actor: str = Field(default="system", min_length=1, max_length=200)
    details: str = Field(default="", max_length=2000)


class AuditEventResponse(BaseModel):
    case_id: str
    timestamp: str
    action: str
    actor: str
    from_status: str | None = None
    to_status: str | None = None
    details: str


class AuditResponse(BaseModel):
    case_id: str
    events: list[AuditEventResponse]
    total: int


class ErrorResponse(BaseModel):
    detail: str