"""Structured evidence context for the investigation copilot.

Constructs a bounded, verified evidence context from RazorGuard
data.  Every field is grounded in actual system data.  No secrets
or credentials are included.

The context is intentionally bounded to avoid sending large
datasets to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class CaseContext:
    """Verified case information."""

    case_id: str
    transaction_id: str
    status: str
    decision: str
    risk_score: float
    risk_level: str
    model_probability: float
    network_score: float
    primary_reason: str
    evidence_text: str
    assigned_to: str
    investigation_narrative: str


@dataclass(frozen=True)
class EvidenceContext:
    """Prioritized evidence items for the LLM."""

    title: str
    severity: str
    category: str
    tier: str
    explanation: str
    investigative_relevance: str


@dataclass(frozen=True)
class NetworkContext:
    """Network intelligence for the transaction."""

    account_id: str
    device_id: str
    merchant_id: str
    account_history_count: int
    related_transaction_count: int
    accounts_on_device: int
    accounts_at_merchant: int
    device_shared: bool
    merchant_shared: bool
    new_device_for_account: bool
    new_merchant_for_account: bool


@dataclass(frozen=True)
class ClusterContext:
    """Coordinated-risk cluster intelligence."""

    cluster_type: str
    risk_score: float
    account_count: int
    device_count: int
    merchant_count: int
    transaction_count: int
    signals: list[str]
    evidence_statements: list[str]


@dataclass(frozen=True)
class InvestigationPathContext:
    """Recommended next investigation steps."""

    title: str
    reason: str
    priority: int


@dataclass(frozen=True)
class AuditContext:
    """Relevant audit events."""

    action: str
    actor: str
    timestamp: str
    details: str


@dataclass(frozen=True)
class CopilotEvidenceContext:
    """Complete bounded evidence context for the LLM."""

    case: CaseContext
    evidence: list[EvidenceContext]
    network: NetworkContext | None
    cluster: ClusterContext | None
    investigation_path: list[InvestigationPathContext]
    audit_events: list[AuditContext]
    evidence_summary: dict[str, int]


# ============================================================
# CONTEXT BUILDER
# ============================================================


_MAX_EVIDENCE_ITEMS = 15
_MAX_AUDIT_EVENTS = 10
_MAX_INVESTIGATION_STEPS = 8


def build_copilot_context(
    *,
    case: dict[str, Any],
    evidence: list[dict[str, Any]],
    network_data: dict[str, Any] | None,
    cluster_data: dict[str, Any] | None,
    investigation_steps: list[dict[str, Any]],
    audit_events: list[dict[str, Any]],
    evidence_summary: dict[str, int],
) -> CopilotEvidenceContext:
    """Build a bounded evidence context from verified RazorGuard data.

    All data comes from the existing intelligence pipeline.
    No data is fabricated.
    """

    # Case context
    case_ctx = CaseContext(
        case_id=str(case.get("case_id", "")),
        transaction_id=str(case.get("transaction_id", "")),
        status=str(case.get("status", "OPEN")),
        decision=str(case.get("decision", "")),
        risk_score=float(case.get("risk_score", 0.0)),
        risk_level=str(case.get("risk_level", "MEDIUM")),
        model_probability=float(case.get("model_probability", 0.0)),
        network_score=float(case.get("network_score", 0.0)),
        primary_reason=str(case.get("primary_reason", "")),
        evidence_text=str(case.get("evidence_text", "")),
        assigned_to=str(case.get("assigned_to") or "Unassigned"),
        investigation_narrative=str(case.get("investigation_narrative", "")),
    )

    # Evidence context (bounded)
    evidence_ctx = [
        EvidenceContext(
            title=str(e.get("title", "")),
            severity=str(e.get("severity", "")),
            category=str(e.get("category", "")),
            tier=str(e.get("tier", "")),
            explanation=str(e.get("explanation", "")),
            investigative_relevance=str(e.get("investigative_relevance", "")),
        )
        for e in evidence[:_MAX_EVIDENCE_ITEMS]
    ]

    # Network context
    network_ctx: NetworkContext | None = None

    if network_data is not None:
        signals = network_data.get("network_risk_signals", {})

        network_ctx = NetworkContext(
            account_id=str(network_data.get("account_id", "")),
            device_id=str(network_data.get("device_id", "")),
            merchant_id=str(network_data.get("merchant_id", "")),
            account_history_count=int(network_data.get("account_history_count", 0)),
            related_transaction_count=int(network_data.get("related_transaction_count", 0)),
            accounts_on_device=len(network_data.get("accounts_seen_on_device", [])),
            accounts_at_merchant=len(network_data.get("accounts_seen_at_merchant", [])),
            device_shared=bool(signals.get("device_shared", False)),
            merchant_shared=bool(signals.get("merchant_shared", False)),
            new_device_for_account=bool(signals.get("new_device_for_account", False)),
            new_merchant_for_account=bool(signals.get("new_merchant_for_account", False)),
        )

    # Cluster context
    cluster_ctx: ClusterContext | None = None

    if cluster_data is not None:
        cluster_ctx = ClusterContext(
            cluster_type=str(cluster_data.get("cluster_type", "")),
            risk_score=float(cluster_data.get("risk_score", 0.0)),
            account_count=len(cluster_data.get("accounts", [])),
            device_count=len(cluster_data.get("devices", [])),
            merchant_count=len(cluster_data.get("merchants", [])),
            transaction_count=len(cluster_data.get("transactions", [])),
            signals=[
                str(s.get("evidence", ""))
                for s in cluster_data.get("signals", [])
            ],
            evidence_statements=[
                str(e) for e in cluster_data.get("evidence", [])
            ],
        )

    # Investigation path (bounded)
    path_ctx = [
        InvestigationPathContext(
            title=str(s.get("title", "")),
            reason=str(s.get("reason", "")),
            priority=int(s.get("priority", 99)),
        )
        for s in investigation_steps[:_MAX_INVESTIGATION_STEPS]
    ]

    # Audit events (bounded, most recent first)
    audit_ctx = [
        AuditContext(
            action=str(e.get("action", "")),
            actor=str(e.get("actor", "")),
            timestamp=str(e.get("timestamp", "")),
            details=str(e.get("details", "")),
        )
        for e in audit_events[:_MAX_AUDIT_EVENTS]
    ]

    return CopilotEvidenceContext(
        case=case_ctx,
        evidence=evidence_ctx,
        network=network_ctx,
        cluster=cluster_ctx,
        investigation_path=path_ctx,
        audit_events=audit_ctx,
        evidence_summary=evidence_summary,
    )


def context_to_prompt(ctx: CopilotEvidenceContext) -> str:
    """Serialize the evidence context into a bounded text block
    suitable for inclusion in an LLM prompt."""

    lines: list[str] = []

    # Case
    lines.append("=== CASE ===")
    lines.append(f"Case ID: {ctx.case.case_id}")
    lines.append(f"Transaction: {ctx.case.transaction_id}")
    lines.append(f"Status: {ctx.case.status}")
    lines.append(f"Decision: {ctx.case.decision}")
    lines.append(f"Risk Score: {ctx.case.risk_score:.1f} ({ctx.case.risk_level})")
    lines.append(f"Model Probability: {ctx.case.model_probability:.1%}")
    lines.append(f"Network Score: {ctx.case.network_score:.1f}")
    lines.append(f"Primary Reason: {ctx.case.primary_reason}")
    lines.append(f"Assigned To: {ctx.case.assigned_to}")

    if ctx.case.evidence_text:
        lines.append(f"Evidence: {ctx.case.evidence_text}")

    if ctx.case.investigation_narrative:
        lines.append(f"Narrative: {ctx.case.investigation_narrative}")

    # Evidence
    lines.append("")
    lines.append("=== EVIDENCE (prioritized) ===")

    for item in ctx.evidence:
        lines.append(
            f"[{item.tier}] [{item.severity}] {item.title} ({item.category})"
        )
        lines.append(f"  Explanation: {item.explanation}")
        lines.append(f"  Relevance: {item.investigative_relevance}")

    # Network
    if ctx.network is not None:
        lines.append("")
        lines.append("=== NETWORK INTELLIGENCE ===")
        lines.append(f"Account: {ctx.network.account_id}")
        lines.append(f"Device: {ctx.network.device_id}")
        lines.append(f"Merchant: {ctx.network.merchant_id}")
        lines.append(f"Account history: {ctx.network.account_history_count} transactions")
        lines.append(f"Related transactions: {ctx.network.related_transaction_count}")
        lines.append(f"Accounts on device: {ctx.network.accounts_on_device}")
        lines.append(f"Accounts at merchant: {ctx.network.accounts_at_merchant}")
        lines.append(f"Device shared: {ctx.network.device_shared}")
        lines.append(f"Merchant shared: {ctx.network.merchant_shared}")
        lines.append(f"New device: {ctx.network.new_device_for_account}")
        lines.append(f"New merchant: {ctx.network.new_merchant_for_account}")

    # Cluster
    if ctx.cluster is not None:
        lines.append("")
        lines.append("=== COORDINATED-RISK CLUSTER ===")
        lines.append(f"Type: {ctx.cluster.cluster_type}")
        lines.append(f"Risk score: {ctx.cluster.risk_score:.1f}")
        lines.append(f"Connected accounts: {ctx.cluster.account_count}")
        lines.append(f"Connected devices: {ctx.cluster.device_count}")
        lines.append(f"Connected merchants: {ctx.cluster.merchant_count}")
        lines.append(f"Related transactions: {ctx.cluster.transaction_count}")

        if ctx.cluster.signals:
            lines.append("Signals:")
            for sig in ctx.cluster.signals:
                lines.append(f"  - {sig}")

        if ctx.cluster.evidence_statements:
            lines.append("Evidence:")
            for ev in ctx.cluster.evidence_statements:
                lines.append(f"  - {ev}")

    # Investigation path
    if ctx.investigation_path:
        lines.append("")
        lines.append("=== INVESTIGATION PATH ===")
        for step in ctx.investigation_path:
            lines.append(f"[P{step.priority}] {step.title}: {step.reason}")

    # Audit
    if ctx.audit_events:
        lines.append("")
        lines.append("=== AUDIT HISTORY ===")
        for event in ctx.audit_events:
            lines.append(
                f"[{event.timestamp}] {event.action} by {event.actor}"
            )
            if event.details:
                lines.append(f"  Details: {event.details}")

    return "\n".join(lines)
