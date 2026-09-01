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
    page: int = 1
    page_size: int = 50
    total_pages: int = 1


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

class DashboardSummaryResponse(BaseModel):
    open_cases: int
    critical_cases: int
    high_cases: int
    medium_cases: int
    low_cases: int
    average_risk_score: float
    total_cases: int

class DashboardDistributionItem(BaseModel):
    label: str
    count: int
    percentage: float


class DashboardActivityItem(BaseModel):
    case_id: str
    transaction_id: str
    action: str
    actor: str
    timestamp: str
    details: str


class DashboardQueueItem(BaseModel):
    case_id: str
    transaction_id: str
    priority: str
    risk_score: float
    risk_level: str
    decision: str
    primary_reason: str


class DashboardDistributionResponse(BaseModel):
    items: list[DashboardDistributionItem]
    total: int


class DashboardActivityResponse(BaseModel):
    items: list[DashboardActivityItem]
    total: int


class DashboardQueueResponse(BaseModel):
    items: list[DashboardQueueItem]
    total: int

class NetworkSummaryResponse(BaseModel):
    accounts: int
    devices: int
    merchants: int

    account_device_edges: int
    account_merchant_edges: int
    device_merchant_edges: int


class NetworkRiskSignals(BaseModel):
    device_shared: bool
    merchant_shared: bool
    new_device_for_account: bool
    new_merchant_for_account: bool

class RiskClusterSignal(BaseModel):
    type: str
    severity: str
    value: int
    evidence: str


class RiskClusterTimelineItem(BaseModel):
    transaction_id: str
    timestamp: str
    account_id: str
    device_id: str
    merchant_id: str


class RiskClusterResponse(BaseModel):
    cluster_id: str
    cluster_type: str
    risk_score: float

    accounts: list[str]
    devices: list[str]
    merchants: list[str]
    transactions: list[str]

    signals: list[RiskClusterSignal]
    evidence: list[str]
    timeline: list[RiskClusterTimelineItem]

class NetworkTransactionResponse(BaseModel):
    transaction_id: str
    timestamp: str

    account_id: str
    device_id: str
    merchant_id: str

    account_history_count: int
    accounts_seen_on_device: list[str]
    accounts_seen_at_merchant: list[str]

    related_transaction_count: int

    network_risk_signals: NetworkRiskSignals

class AnalyticsDistributionItem(BaseModel):
    label: str
    count: int
    percentage: float


class AnalyticsMetricResponse(BaseModel):
    total_cases: int
    average_risk_score: float
    median_risk_score: float
    maximum_risk_score: float

    average_model_probability: float
    average_network_score: float

    priority_distribution: list[AnalyticsDistributionItem]
    risk_level_distribution: list[AnalyticsDistributionItem]
    decision_distribution: list[AnalyticsDistributionItem]
    status_distribution: list[AnalyticsDistributionItem]

    top_reasons: list[AnalyticsDistributionItem]


# ============================================================
# INVESTIGATION INTELLIGENCE
# ============================================================


class EvidenceItemResponse(BaseModel):
    title: str
    severity: str
    category: str
    explanation: str
    investigative_relevance: str
    supporting_entities: list[str]
    supporting_transactions: list[str]
    observed_value: str


class PrioritizedEvidenceResponse(BaseModel):
    title: str
    severity: str
    category: str
    explanation: str
    investigative_relevance: str
    supporting_entities: list[str]
    supporting_transactions: list[str]
    observed_value: str
    tier: str
    priority_score: float
    rank: int


class InvestigationStepResponse(BaseModel):
    priority: int
    title: str
    reason: str
    supporting_evidence: list[str]
    target_entity: str
    navigation_target: str


class CaseIntelligenceResponse(BaseModel):
    case_id: str
    transaction_id: str
    risk_score: float
    risk_level: str
    decision: str
    primary_reason: str

    evidence: list[PrioritizedEvidenceResponse]
    investigation_steps: list[InvestigationStepResponse]
    evidence_summary: dict[str, int]