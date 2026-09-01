"""Coordinated-risk evidence synthesis.

Identifies convergence of independent risk signals to produce
structured, explainable evidence items for investigators.

Every evidence item is grounded in actual observable data.
No signals are fabricated or assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class EvidenceItem:
    """A single synthesized evidence item."""

    title: str
    severity: str  # HIGH, MEDIUM, LOW
    category: str  # NETWORK, BEHAVIORAL, TRANSACTION, CLUSTER, CONVERGENCE
    explanation: str
    investigative_relevance: str
    supporting_entities: list[str] = field(default_factory=list)
    supporting_transactions: list[str] = field(default_factory=list)
    observed_value: str = ""


# ============================================================
# EVIDENCE SYNTHESIS
# ============================================================


def synthesize_network_evidence(
    *,
    network_risk_signals: dict[str, Any] | None,
    accounts_seen_on_device: list[str] | None,
    accounts_seen_at_merchant: list[str] | None,
    account_history_count: int,
    related_transaction_count: int,
) -> list[EvidenceItem]:
    """Produce evidence items from network intelligence.

    Uses only observed network signals from the backend
    investigation module. Thresholds match the existing
    investigator.py logic.
    """

    items: list[EvidenceItem] = []

    if network_risk_signals is None:
        return items

    device_shared = bool(
        network_risk_signals.get("device_shared", False)
    )
    merchant_shared = bool(
        network_risk_signals.get("merchant_shared", False)
    )
    new_device = bool(
        network_risk_signals.get(
            "new_device_for_account", False
        )
    )
    new_merchant = bool(
        network_risk_signals.get(
            "new_merchant_for_account", False
        )
    )

    device_accounts = accounts_seen_on_device or []
    merchant_accounts = accounts_seen_at_merchant or []

    # Shared device evidence
    if device_shared and len(device_accounts) >= 2:
        severity = (
            "HIGH"
            if len(device_accounts) >= 3
            else "MEDIUM"
        )

        items.append(
            EvidenceItem(
                title="Device shared across multiple accounts",
                severity=severity,
                category="NETWORK",
                explanation=(
                    f"This device has been used by "
                    f"{len(device_accounts)} different accounts. "
                    f"Shared device usage is a strong indicator "
                    f"of coordinated activity or account compromise."
                ),
                investigative_relevance=(
                    "Review whether the accounts sharing this device "
                    "are related or show coordinated purchasing patterns."
                ),
                supporting_entities=[
                    f"device:{account}"
                    for account in device_accounts[:5]
                ],
                observed_value=f"{len(device_accounts)} accounts",
            )
        )

    # New device evidence
    if new_device:
        items.append(
            EvidenceItem(
                title="New device for this account",
                severity="MEDIUM",
                category="NETWORK",
                explanation=(
                    "This device has not been previously associated "
                    "with this account. New device usage may indicate "
                    "account compromise or legitimate device change."
                ),
                investigative_relevance=(
                    "Cross-reference with the account's recent "
                    "activity to determine if the device change "
                    "is consistent with normal behavior."
                ),
                supporting_entities=[],
                observed_value="First occurrence",
            )
        )

    # Shared merchant evidence
    if merchant_shared and len(merchant_accounts) >= 3:
        severity = (
            "HIGH"
            if len(merchant_accounts) >= 10
            else "MEDIUM"
        )

        items.append(
            EvidenceItem(
                title="Merchant concentration across accounts",
                severity=severity,
                category="NETWORK",
                explanation=(
                    f"This merchant has been transacted with by "
                    f"{len(merchant_accounts)} accounts. High "
                    f"merchant concentration may indicate "
                    f"coordinated purchasing or merchant-side fraud."
                ),
                investigative_relevance=(
                    "Review whether the accounts at this merchant "
                    "share common attributes or show coordinated "
                    "transaction timing."
                ),
                supporting_entities=[
                    f"merchant:{account}"
                    for account in merchant_accounts[:5]
                ],
                observed_value=f"{len(merchant_accounts)} accounts",
            )
        )

    # New merchant evidence
    if new_merchant:
        items.append(
            EvidenceItem(
                title="New merchant relationship",
                severity="LOW",
                category="NETWORK",
                explanation=(
                    "This account has not previously transacted "
                    "with this merchant. New merchant relationships "
                    "combined with other risk signals warrant review."
                ),
                investigative_relevance=(
                    "Compare with the account's established "
                    "merchant history to assess consistency."
                ),
                supporting_entities=[],
                observed_value="First occurrence",
            )
        )

    # Account history context
    if account_history_count == 0:
        items.append(
            EvidenceItem(
                title="No prior account history",
                severity="MEDIUM",
                category="NETWORK",
                explanation=(
                    "This account has no prior transaction history "
                    "before this event. New accounts are "
                    "disproportionately associated with fraud."
                ),
                investigative_relevance=(
                    "Review account creation details and any "
                    "available registration signals."
                ),
                supporting_entities=[],
                observed_value="0 prior transactions",
            )
        )

    # Related transaction volume
    if related_transaction_count >= 5:
        items.append(
            EvidenceItem(
                title="High related transaction volume",
                severity="MEDIUM",
                category="NETWORK",
                explanation=(
                    f"{related_transaction_count} transactions are "
                    f"connected to this account through shared "
                    f"devices or merchants. A high volume of "
                    f"related transactions increases coordination risk."
                ),
                investigative_relevance=(
                    "Review the related transaction set for "
                    "temporal patterns, amount anomalies, or "
                    "entity overlap."
                ),
                supporting_entities=[],
                observed_value=f"{related_transaction_count} related transactions",
            )
        )

    return items


def synthesize_cluster_evidence(
    *,
    cluster_signals: list[dict[str, Any]] | None,
    cluster_evidence: list[str] | None,
    cluster_type: str | None,
    cluster_risk_score: float | None,
    cluster_accounts: list[str] | None,
    cluster_devices: list[str] | None,
    cluster_merchants: list[str] | None,
    cluster_transactions: list[str] | None,
) -> list[EvidenceItem]:
    """Produce evidence items from coordinated-risk cluster.

    Uses the existing cluster signal types from clusters.py.
    """

    items: list[EvidenceItem] = []

    if not cluster_signals:
        return items

    # Map cluster signal types to evidence items
    signal_severity_map = {
        "SHARED_DEVICE": "HIGH",
        "MULTI_ACCOUNT_CONNECTION": "MEDIUM",
        "SHARED_MERCHANT": "MEDIUM",
        "TRANSACTION_CLUSTER": "MEDIUM",
        "TEMPORAL_BURST": "HIGH",
    }

    for signal in cluster_signals:
        signal_type = signal.get("type", "")
        severity = signal.get(
            "severity",
            signal_severity_map.get(signal_type, "MEDIUM"),
        )
        value = signal.get("value", 0)
        evidence_text = signal.get("evidence", "")

        if signal_type == "SHARED_DEVICE":
            items.append(
                EvidenceItem(
                    title="Coordinated device usage",
                    severity=severity,
                    category="CLUSTER",
                    explanation=(
                        f"{evidence_text}. Multiple accounts "
                        f"are using the same device, which is "
                        f"a strong coordination signal."
                    ),
                    investigative_relevance=(
                        "This is a primary coordination indicator. "
                        "Review all accounts on this device for "
                        "related transaction patterns."
                    ),
                    supporting_entities=(
                        [f"device:{d}" for d in (cluster_devices or [])[:3]]
                    ),
                    supporting_transactions=(
                        (cluster_transactions or [])[:5]
                    ),
                    observed_value=evidence_text,
                )
            )
        elif signal_type == "MULTI_ACCOUNT_CONNECTION":
            items.append(
                EvidenceItem(
                    title="Multi-account coordination",
                    severity=severity,
                    category="CLUSTER",
                    explanation=(
                        f"{evidence_text}. Accounts are connected "
                        f"through shared devices or merchants."
                    ),
                    investigative_relevance=(
                        "Review account relationships and "
                        "transaction timing across the connected group."
                    ),
                    supporting_entities=(
                        [f"account:{a}" for a in (cluster_accounts or [])[:5]]
                    ),
                    supporting_transactions=(
                        (cluster_transactions or [])[:5]
                    ),
                    observed_value=f"{value} connected accounts",
                )
            )
        elif signal_type == "SHARED_MERCHANT":
            items.append(
                EvidenceItem(
                    title="Merchant convergence",
                    severity=severity,
                    category="CLUSTER",
                    explanation=(
                        f"{evidence_text}. Multiple accounts "
                        f"are transacting with the same merchant."
                    ),
                    investigative_relevance=(
                        "Review merchant-side activity for "
                        "coordinated purchasing patterns."
                    ),
                    supporting_entities=(
                        [f"merchant:{m}" for m in (cluster_merchants or [])[:3]]
                    ),
                    supporting_transactions=(
                        (cluster_transactions or [])[:5]
                    ),
                    observed_value=evidence_text,
                )
            )
        elif signal_type == "TEMPORAL_BURST":
            items.append(
                EvidenceItem(
                    title="Temporal transaction burst",
                    severity=severity,
                    category="CLUSTER",
                    explanation=(
                        f"{evidence_text}. Multiple transactions "
                        f"occurred within a short time window."
                    ),
                    investigative_relevance=(
                        "Temporal bursts are strong coordination "
                        "evidence. Review transaction timestamps "
                        "and amounts for automated patterns."
                    ),
                    supporting_entities=(
                        [f"account:{a}" for a in (cluster_accounts or [])[:3]]
                    ),
                    supporting_transactions=(
                        (cluster_transactions or [])[:5]
                    ),
                    observed_value=evidence_text,
                )
            )
        elif signal_type == "TRANSACTION_CLUSTER":
            items.append(
                EvidenceItem(
                    title="Transaction clustering",
                    severity=severity,
                    category="CLUSTER",
                    explanation=(
                        f"{evidence_text}. Related transactions "
                        f"form a coordination cluster."
                    ),
                    investigative_relevance=(
                        "Review the transaction cluster for "
                        "amount patterns, merchant overlap, "
                        "or device sharing."
                    ),
                    supporting_transactions=(
                        (cluster_transactions or [])[:5]
                    ),
                    observed_value=f"{value} related transactions",
                )
            )

    # Cluster-level evidence statements
    if cluster_type == "COORDINATED_NETWORK" and cluster_risk_score and cluster_risk_score >= 40:
        items.append(
            EvidenceItem(
                title="Coordinated network detected",
                severity="HIGH",
                category="CONVERGENCE",
                explanation=(
                    f"The risk cluster has been classified as "
                    f"COORDINATED_NETWORK with a risk score of "
                    f"{cluster_risk_score:.1f}. This indicates "
                    f"multi-entity coordination across devices, "
                    f"merchants, or temporal patterns."
                ),
                investigative_relevance=(
                    "This is the strongest available indicator "
                    "of coordinated fraud. Prioritize review of "
                    "all connected entities."
                ),
                supporting_entities=(
                    [f"account:{a}" for a in (cluster_accounts or [])[:5]]
                    + [f"device:{d}" for d in (cluster_devices or [])[:3]]
                ),
                supporting_transactions=(
                    (cluster_transactions or [])[:10]
                ),
                observed_value=f"score={cluster_risk_score:.1f}",
            )
        )

    return items


def synthesize_convergence_evidence(
    *,
    network_items: list[EvidenceItem],
    cluster_items: list[EvidenceItem],
    risk_score: float,
    model_probability: float,
    network_score: float,
) -> list[EvidenceItem]:
    """Detect convergence of independent signals.

    When multiple independent evidence categories converge
    on the same conclusion, this produces a CONVERGENCE
    evidence item explaining why the case is high-risk.

    Convergence is defined as:
    - At least one NETWORK evidence item, AND
    - At least one CLUSTER evidence item, OR
    - Risk score >= 70 (HIGH), OR
    - Both model probability >= 0.70 AND network score >= 7.0
    """

    items: list[EvidenceItem] = []

    has_network = any(
        e.category == "NETWORK" for e in network_items
    )
    has_cluster = any(
        e.category == "CLUSTER" for e in cluster_items
    )

    convergence_count = sum([
        has_network,
        has_cluster,
        risk_score >= 70,
        model_probability >= 0.70 and network_score >= 7.0,
    ])

    if convergence_count >= 2:
        sources: list[str] = []

        if has_network:
            sources.append("network intelligence")

        if has_cluster:
            sources.append("coordinated-risk cluster")

        if risk_score >= 70:
            sources.append(
                f"elevated risk score ({risk_score:.1f})"
            )

        if model_probability >= 0.70 and network_score >= 7.0:
            sources.append(
                f"model probability ({model_probability:.0%}) "
                f"and network risk ({network_score:.1f})"
            )

        severity = (
            "HIGH"
            if risk_score >= 70
            else "MEDIUM"
        )

        items.append(
            EvidenceItem(
                title="Converging risk indicators",
                severity=severity,
                category="CONVERGENCE",
                explanation=(
                    f"Multiple independent risk indicators converge: "
                    f"{', '.join(sources)}. When independent evidence "
                    f"channels agree, the overall risk assessment "
                    f"is more reliable."
                ),
                investigative_relevance=(
                    "Converging evidence is the strongest basis "
                    "for investigation prioritization. Review "
                    "each evidence source to understand the "
                    "full risk picture."
                ),
                observed_value=f"{convergence_count} independent signals",
            )
        )

    return items


def build_coordinated_evidence(
    *,
    network_risk_signals: dict[str, Any] | None,
    accounts_seen_on_device: list[str] | None,
    accounts_seen_at_merchant: list[str] | None,
    account_history_count: int,
    related_transaction_count: int,
    cluster_signals: list[dict[str, Any]] | None,
    cluster_evidence: list[str] | None,
    cluster_type: str | None,
    cluster_risk_score: float | None,
    cluster_accounts: list[str] | None,
    cluster_devices: list[str] | None,
    cluster_merchants: list[str] | None,
    cluster_transactions: list[str] | None,
    risk_score: float,
    model_probability: float,
    network_score: float,
) -> list[EvidenceItem]:
    """Master synthesis function.

    Combines network evidence, cluster evidence, and
    convergence detection into a unified evidence list.
    """

    network_items = synthesize_network_evidence(
        network_risk_signals=network_risk_signals,
        accounts_seen_on_device=accounts_seen_on_device,
        accounts_seen_at_merchant=accounts_seen_at_merchant,
        account_history_count=account_history_count,
        related_transaction_count=related_transaction_count,
    )

    cluster_items = synthesize_cluster_evidence(
        cluster_signals=cluster_signals,
        cluster_evidence=cluster_evidence,
        cluster_type=cluster_type,
        cluster_risk_score=cluster_risk_score,
        cluster_accounts=cluster_accounts,
        cluster_devices=cluster_devices,
        cluster_merchants=cluster_merchants,
        cluster_transactions=cluster_transactions,
    )

    convergence_items = synthesize_convergence_evidence(
        network_items=network_items,
        cluster_items=cluster_items,
        risk_score=risk_score,
        model_probability=model_probability,
        network_score=network_score,
    )

    return convergence_items + cluster_items + network_items


def evidence_to_dict(item: EvidenceItem) -> dict[str, Any]:
    """Convert an EvidenceItem to a JSON-serializable dict."""
    return {
        "title": item.title,
        "severity": item.severity,
        "category": item.category,
        "explanation": item.explanation,
        "investigative_relevance": item.investigative_relevance,
        "supporting_entities": item.supporting_entities,
        "supporting_transactions": item.supporting_transactions,
        "observed_value": item.observed_value,
    }
