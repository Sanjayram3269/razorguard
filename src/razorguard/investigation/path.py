"""Deterministic investigation path engine.

Inspects available evidence and case state to produce
actionable next-best-investigation-step recommendations.

Each step is grounded in actual available evidence.
No steps are shown when the underlying evidence is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from razorguard.graph.evidence import EvidenceItem
from razorguard.graph.prioritization import (
    PrioritizedEvidence,
    prioritize_evidence,
)


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class InvestigationStep:
    """A single recommended investigation step."""

    priority: int  # 1 = highest
    title: str
    reason: str
    supporting_evidence: list[str] = field(default_factory=list)
    target_entity: str = ""  # e.g. "device", "merchant", "account"
    navigation_target: str = ""  # e.g. "network", "cases"


# ============================================================
# STEP GENERATORS
# ============================================================


def _steps_from_network_evidence(
    *,
    device_shared: bool,
    merchant_shared: bool,
    new_device: bool,
    accounts_on_device: int,
    accounts_at_merchant: int,
) -> list[InvestigationStep]:
    """Generate investigation steps from network evidence."""

    steps: list[InvestigationStep] = []

    if device_shared and accounts_on_device >= 2:
        steps.append(
            InvestigationStep(
                priority=2,
                title="Review device-connected accounts",
                reason=(
                    f"This device is shared across "
                    f"{accounts_on_device} accounts. Review "
                    f"whether these accounts show coordinated "
                    f"activity patterns."
                ),
                supporting_evidence=[
                    "Device shared across multiple accounts"
                ],
                target_entity="device",
                navigation_target="network",
            )
        )

    if merchant_shared and accounts_at_merchant >= 3:
        steps.append(
            InvestigationStep(
                priority=3,
                title="Inspect merchant-connected accounts",
                reason=(
                    f"This merchant has been used by "
                    f"{accounts_at_merchant} accounts. Review "
                    f"for coordinated purchasing patterns."
                ),
                supporting_evidence=[
                    "Merchant concentration across accounts"
                ],
                target_entity="merchant",
                navigation_target="network",
            )
        )

    if new_device:
        steps.append(
            InvestigationStep(
                priority=3,
                title="Review account history",
                reason=(
                    "This device is new for this account. "
                    "Review the account's prior transaction "
                    "history and device usage patterns."
                ),
                supporting_evidence=[
                    "New device for account"
                ],
                target_entity="account",
                navigation_target="network",
            )
        )

    return steps


def _steps_from_cluster_evidence(
    *,
    cluster_type: str | None,
    cluster_risk_score: float | None,
    cluster_accounts: list[str] | None,
    cluster_devices: list[str] | None,
    cluster_transactions: list[str] | None,
    has_temporal_burst: bool,
) -> list[InvestigationStep]:
    """Generate investigation steps from cluster evidence."""

    steps: list[InvestigationStep] = []

    if (
        cluster_type == "COORDINATED_NETWORK"
        and cluster_risk_score
        and cluster_risk_score >= 40
    ):
        steps.append(
            InvestigationStep(
                priority=1,
                title="Examine coordinated-risk cluster",
                reason=(
                    f"A COORDINATED_NETWORK cluster was detected "
                    f"with risk score {cluster_risk_score:.1f}. "
                    f"This is the strongest indicator of "
                    f"multi-entity coordination."
                ),
                supporting_evidence=[
                    f"{len(cluster_accounts or [])} connected accounts",
                    f"{len(cluster_devices or [])} connected devices",
                ],
                target_entity="cluster",
                navigation_target="network",
            )
        )

    if has_temporal_burst:
        steps.append(
            InvestigationStep(
                priority=2,
                title="Investigate temporal burst pattern",
                reason=(
                    "Multiple transactions occurred within a "
                    "short time window. Review transaction "
                    "timestamps and amounts for automated patterns."
                ),
                supporting_evidence=[
                    "Temporal transaction burst detected"
                ],
                target_entity="transactions",
                navigation_target="network",
            )
        )

    if cluster_transactions and len(cluster_transactions) >= 5:
        steps.append(
            InvestigationStep(
                priority=3,
                title="Review related transaction set",
                reason=(
                    f"{len(cluster_transactions)} related transactions "
                    f"were identified. Review for amount patterns, "
                    f"merchant overlap, or device sharing."
                ),
                supporting_evidence=[
                    f"{len(cluster_transactions)} related transactions"
                ],
                target_entity="transactions",
                navigation_target="network",
            )
        )

    return steps


def _steps_from_case_state(
    *,
    status: str,
    risk_score: float,
    decision: str,
    model_probability: float,
    network_score: float,
    assigned_to: str | None,
    has_audit_events: bool,
) -> list[InvestigationStep]:
    """Generate investigation steps based on case state."""

    steps: list[InvestigationStep] = []

    if status == "OPEN" and not assigned_to:
        steps.append(
            InvestigationStep(
                priority=1,
                title="Assign case to investigator",
                reason=(
                    "This case is open and unassigned. "
                    "Assign to an investigator for review."
                ),
                supporting_evidence=[],
                target_entity="case",
                navigation_target="cases",
            )
        )

    if status == "OPEN" and assigned_to:
        steps.append(
            InvestigationStep(
                priority=2,
                title="Begin investigation review",
                reason=(
                    f"Case is assigned to {assigned_to} "
                    f"and awaiting review. Start the "
                    f"investigation workflow."
                ),
                supporting_evidence=[],
                target_entity="case",
                navigation_target="cases",
            )
        )

    if decision == "BLOCK":
        steps.append(
            InvestigationStep(
                priority=1,
                title="Review high-risk decision",
                reason=(
                    f"This transaction was BLOCKED with a "
                    f"risk score of {risk_score:.1f}. "
                    f"Review the block decision and supporting "
                    f"evidence."
                ),
                supporting_evidence=[
                    f"Decision: BLOCK",
                    f"Risk score: {risk_score:.1f}",
                ],
                target_entity="case",
                navigation_target="cases",
            )
        )

    if model_probability >= 0.80:
        steps.append(
            InvestigationStep(
                priority=2,
                title="Review model prediction",
                reason=(
                    f"Model probability is {model_probability:.0%}, "
                    f"indicating high estimated risk. Review the "
                    f"model's reasoning and supporting features."
                ),
                supporting_evidence=[
                    f"Model probability: {model_probability:.0%}"
                ],
                target_entity="model",
                navigation_target="analytics",
            )
        )

    if network_score >= 7.0:
        steps.append(
            InvestigationStep(
                priority=2,
                title="Investigate network risk",
                reason=(
                    f"Network risk score is {network_score:.1f}, "
                    f"indicating significant entity relationship "
                    f"risk. Review network intelligence."
                ),
                supporting_evidence=[
                    f"Network score: {network_score:.1f}"
                ],
                target_entity="network",
                navigation_target="network",
            )
        )

    if has_audit_events:
        steps.append(
            InvestigationStep(
                priority=4,
                title="Review case audit history",
                reason=(
                    "Review the full audit trail to understand "
                    "previous investigation actions and outcomes."
                ),
                supporting_evidence=[],
                target_entity="audit",
                navigation_target="cases",
            )
        )

    return steps


# ============================================================
# MASTER FUNCTION
# ============================================================


def build_investigation_path(
    *,
    status: str,
    risk_score: float,
    decision: str,
    model_probability: float,
    network_score: float,
    assigned_to: str | None,
    has_audit_events: bool,
    # Network intelligence
    device_shared: bool,
    merchant_shared: bool,
    new_device: bool,
    accounts_on_device: int,
    accounts_at_merchant: int,
    # Cluster intelligence
    cluster_type: str | None = None,
    cluster_risk_score: float | None = None,
    cluster_accounts: list[str] | None = None,
    cluster_devices: list[str] | None = None,
    cluster_transactions: list[str] | None = None,
    has_temporal_burst: bool = False,
) -> list[InvestigationStep]:
    """Build the recommended investigation path.

    Returns a prioritized list of actionable investigation steps.
    Steps are deduplicated and sorted by priority (1=highest).
    """

    all_steps: list[InvestigationStep] = []

    all_steps.extend(
        _steps_from_case_state(
            status=status,
            risk_score=risk_score,
            decision=decision,
            model_probability=model_probability,
            network_score=network_score,
            assigned_to=assigned_to,
            has_audit_events=has_audit_events,
        )
    )

    all_steps.extend(
        _steps_from_network_evidence(
            device_shared=device_shared,
            merchant_shared=merchant_shared,
            new_device=new_device,
            accounts_on_device=accounts_on_device,
            accounts_at_merchant=accounts_at_merchant,
        )
    )

    all_steps.extend(
        _steps_from_cluster_evidence(
            cluster_type=cluster_type,
            cluster_risk_score=cluster_risk_score,
            cluster_accounts=cluster_accounts,
            cluster_devices=cluster_devices,
            cluster_transactions=cluster_transactions,
            has_temporal_burst=has_temporal_burst,
        )
    )

    # Deduplicate by title
    seen_titles: set[str] = set()
    unique_steps: list[InvestigationStep] = []

    for step in all_steps:
        if step.title not in seen_titles:
            seen_titles.add(step.title)
            unique_steps.append(step)

    # Sort by priority ascending (1 = highest)
    unique_steps.sort(key=lambda s: s.priority)

    return unique_steps


def step_to_dict(step: InvestigationStep) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
        "priority": step.priority,
        "title": step.title,
        "reason": step.reason,
        "supporting_evidence": step.supporting_evidence,
        "target_entity": step.target_entity,
        "navigation_target": step.navigation_target,
    }
