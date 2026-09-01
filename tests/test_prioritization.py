"""Tests for evidence prioritization."""
from __future__ import annotations

import pytest

from razorguard.graph.evidence import EvidenceItem
from razorguard.graph.prioritization import (
    PrioritizedEvidence,
    group_by_tier,
    prioritize_evidence,
    prioritized_to_dict,
)


def _make_item(
    title: str = "Test",
    severity: str = "MEDIUM",
    category: str = "NETWORK",
    entities: int = 0,
    transactions: int = 0,
) -> EvidenceItem:
    return EvidenceItem(
        title=title,
        severity=severity,
        category=category,
        explanation="explanation",
        investigative_relevance="relevance",
        supporting_entities=[f"e{i}" for i in range(entities)],
        supporting_transactions=[f"t{i}" for i in range(transactions)],
        observed_value="value",
    )


# ============================================================
# SCORING
# ============================================================


class TestScoring:
    def test_high_cluster_scores_higher_than_medium_network(self):
        high_cluster = _make_item(
            severity="HIGH",
            category="CLUSTER",
        )
        medium_network = _make_item(
            severity="MEDIUM",
            category="NETWORK",
        )

        prioritized = prioritize_evidence(
            [medium_network, high_cluster]
        )

        assert prioritized[0].item.category == "CLUSTER"
        assert prioritized[0].tier == "PRIMARY"

    def test_convergence_scores_highest(self):
        convergence = _make_item(
            severity="HIGH",
            category="CONVERGENCE",
        )
        cluster = _make_item(
            severity="HIGH",
            category="CLUSTER",
        )
        network = _make_item(
            severity="MEDIUM",
            category="NETWORK",
        )

        prioritized = prioritize_evidence(
            [network, cluster, convergence]
        )

        assert prioritized[0].item.category == "CONVERGENCE"
        assert prioritized[0].tier == "PRIMARY"

    def test_entity_bonus_increases_score(self):
        low_entities = _make_item(
            severity="MEDIUM",
            category="NETWORK",
            entities=0,
        )
        high_entities = _make_item(
            title="With entities",
            severity="MEDIUM",
            category="NETWORK",
            entities=5,
        )

        prioritized = prioritize_evidence(
            [low_entities, high_entities]
        )

        assert prioritized[0].item.title == "With entities"


# ============================================================
# TIER ASSIGNMENT
# ============================================================


class TestTierAssignment:
    def test_primary_tier(self):
        item = _make_item(
            severity="HIGH",
            category="CONVERGENCE",
        )

        prioritized = prioritize_evidence([item])

        assert prioritized[0].tier == "PRIMARY"

    def test_supporting_tier(self):
        item = _make_item(
            severity="MEDIUM",
            category="NETWORK",
        )

        prioritized = prioritize_evidence([item])

        # MEDIUM NETWORK = 2 * 2 = 4, which is SUPPORTING
        assert prioritized[0].tier in (
            "PRIMARY",
            "SUPPORTING",
        )

    def test_contextual_tier(self):
        item = _make_item(
            severity="LOW",
            category="TRANSACTION",
        )

        prioritized = prioritize_evidence([item])

        # LOW TRANSACTION = 1.5 * 1 = 1.5, which is CONTEXTUAL
        assert prioritized[0].tier == "CONTEXTUAL"


# ============================================================
# GROUPING
# ============================================================


class TestGrouping:
    def test_group_by_tier(self):
        items = [
            _make_item(
                title="Primary",
                severity="HIGH",
                category="CONVERGENCE",
            ),
            _make_item(
                title="Contextual",
                severity="LOW",
                category="TRANSACTION",
            ),
        ]

        prioritized = prioritize_evidence(items)
        groups = group_by_tier(prioritized)

        assert "PRIMARY" in groups
        assert "SUPPORTING" in groups
        assert "CONTEXTUAL" in groups

        assert len(groups["PRIMARY"]) >= 1
        assert len(groups["CONTEXTUAL"]) >= 1


# ============================================================
# EDGE CASES
# ============================================================


class TestEdgeCases:
    def test_empty_list(self):
        prioritized = prioritize_evidence([])

        assert prioritized == []

    def test_single_item(self):
        item = _make_item()

        prioritized = prioritize_evidence([item])

        assert len(prioritized) == 1
        assert prioritized[0].rank == 1

    def test_sorting_by_score(self):
        items = [
            _make_item(
                title="Low",
                severity="LOW",
                category="TRANSACTION",
            ),
            _make_item(
                title="High",
                severity="HIGH",
                category="CONVERGENCE",
            ),
            _make_item(
                title="Medium",
                severity="MEDIUM",
                category="CLUSTER",
            ),
        ]

        prioritized = prioritize_evidence(items)

        assert prioritized[0].item.title == "High"
        assert prioritized[1].item.title == "Medium"
        assert prioritized[2].item.title == "Low"


# ============================================================
# TO DICT
# ============================================================


class TestToDict:
    def test_roundtrip(self):
        item = _make_item()

        prioritized = prioritize_evidence([item])
        d = prioritized_to_dict(prioritized[0])

        assert d["title"] == "Test"
        assert "tier" in d
        assert "priority_score" in d
        assert "rank" in d
