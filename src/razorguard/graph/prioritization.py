"""Deterministic evidence prioritization.

Groups synthesized evidence into three tiers:
    PRIMARY   — strongest reasons for the case
    SUPPORTING — corroborating evidence
    CONTEXTUAL — background information

Prioritization is explainable and documented below.

SCORING RULES:

Each evidence item receives a priority score based on:
  1. Category weight (CONVERGENCE > CLUSTER > NETWORK)
  2. Severity multiplier (HIGH=3, MEDIUM=2, LOW=1)
  3. Supporting entity/transaction count bonus

TIER THRESHOLDS:
  PRIMARY:    score >= 8
  SUPPORTING: score >= 4
  CONTEXTUAL: score < 4

This is intentionally simple and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from razorguard.graph.evidence import EvidenceItem


# ============================================================
# SCORING
# ============================================================


CATEGORY_WEIGHT: dict[str, float] = {
    "CONVERGENCE": 5.0,
    "CLUSTER": 3.0,
    "NETWORK": 2.0,
    "BEHAVIORAL": 2.0,
    "TRANSACTION": 1.5,
}

SEVERITY_MULTIPLIER: dict[str, float] = {
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
}


def _score_item(item: EvidenceItem) -> float:
    """Compute a deterministic priority score.

    Score = category_weight * severity_multiplier + entity_bonus

    entity_bonus = min(len(supporting_entities) * 0.3, 2.0)
                  + min(len(supporting_transactions) * 0.2, 2.0)

    Caps:
    - Maximum entity bonus is 4.0
    - Maximum possible score is ~19.0
    """

    category_weight = CATEGORY_WEIGHT.get(
        item.category, 1.0
    )

    severity_multiplier = SEVERITY_MULTIPLIER.get(
        item.severity, 1.0
    )

    base_score = category_weight * severity_multiplier

    entity_bonus = min(
        len(item.supporting_entities) * 0.3, 2.0
    )

    transaction_bonus = min(
        len(item.supporting_transactions) * 0.2, 2.0
    )

    return base_score + entity_bonus + transaction_bonus


# ============================================================
# TIER ASSIGNMENT
# ============================================================


PRIMARY_THRESHOLD = 8.0
SUPPORTING_THRESHOLD = 4.0


def _assign_tier(score: float) -> str:
    """Assign priority tier based on score."""

    if score >= PRIMARY_THRESHOLD:
        return "PRIMARY"

    if score >= SUPPORTING_THRESHOLD:
        return "SUPPORTING"

    return "CONTEXTUAL"


# ============================================================
# PUBLIC API
# ============================================================


@dataclass(frozen=True)
class PrioritizedEvidence:
    """An evidence item with assigned priority tier and score."""

    item: EvidenceItem
    tier: str
    score: float
    rank: int


def prioritize_evidence(
    items: list[EvidenceItem],
) -> list[PrioritizedEvidence]:
    """Prioritize evidence items into tiers.

    Returns items sorted by score descending, each tagged
    with its tier (PRIMARY, SUPPORTING, or CONTEXTUAL).
    """

    scored = [
        (item, _score_item(item))
        for item in items
    ]

    scored.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    result: list[PrioritizedEvidence] = []

    for rank, (item, score) in enumerate(scored, start=1):
        tier = _assign_tier(score)

        result.append(
            PrioritizedEvidence(
                item=item,
                tier=tier,
                score=round(score, 2),
                rank=rank,
            )
        )

    return result


def group_by_tier(
    prioritized: list[PrioritizedEvidence],
) -> dict[str, list[PrioritizedEvidence]]:
    """Group prioritized evidence by tier."""

    groups: dict[str, list[PrioritizedEvidence]] = {
        "PRIMARY": [],
        "SUPPORTING": [],
        "CONTEXTUAL": [],
    }

    for item in prioritized:
        groups[item.tier].append(item)

    return groups


def prioritized_to_dict(item: PrioritizedEvidence) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    base = {
        "title": item.item.title,
        "severity": item.item.severity,
        "category": item.item.category,
        "explanation": item.item.explanation,
        "investigative_relevance": item.item.investigative_relevance,
        "supporting_entities": item.item.supporting_entities,
        "supporting_transactions": item.item.supporting_transactions,
        "observed_value": item.item.observed_value,
        "tier": item.tier,
        "priority_score": item.score,
        "rank": item.rank,
    }
    return base
