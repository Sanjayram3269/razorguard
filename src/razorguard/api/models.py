from __future__ import annotations

from datetime import datetime

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


class ErrorResponse(BaseModel):
    detail: str