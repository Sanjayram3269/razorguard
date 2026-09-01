"""Tests for coordinated-risk evidence synthesis."""
from __future__ import annotations

import pytest

from razorguard.graph.evidence import (
    EvidenceItem,
    build_coordinated_evidence,
    evidence_to_dict,
    synthesize_cluster_evidence,
    synthesize_convergence_evidence,
    synthesize_network_evidence,
)


# ============================================================
# NETWORK EVIDENCE
# ============================================================


class TestNetworkEvidence:
    def test_shared_device_high_severity(self):
        items = synthesize_network_evidence(
            network_risk_signals={"device_shared": True},
            accounts_seen_on_device=["A", "B", "C"],
            accounts_seen_at_merchant=[],
            account_history_count=10,
            related_transaction_count=5,
        )

        titles = [i.title for i in items]

        assert "Device shared across multiple accounts" in titles

        shared = [
            i for i in items
            if i.title == "Device shared across multiple accounts"
        ]

        assert len(shared) == 1
        assert shared[0].severity == "HIGH"
        assert shared[0].category == "NETWORK"

    def test_shared_device_medium_severity(self):
        items = synthesize_network_evidence(
            network_risk_signals={"device_shared": True},
            accounts_seen_on_device=["A", "B"],
            accounts_seen_at_merchant=[],
            account_history_count=10,
            related_transaction_count=5,
        )

        shared = [
            i for i in items
            if i.title == "Device shared across multiple accounts"
        ]

        assert len(shared) == 1
        assert shared[0].severity == "MEDIUM"

    def test_new_device(self):
        items = synthesize_network_evidence(
            network_risk_signals={"new_device_for_account": True},
            accounts_seen_on_device=[],
            accounts_seen_at_merchant=[],
            account_history_count=5,
            related_transaction_count=2,
        )

        titles = [i.title for i in items]

        assert "New device for this account" in titles

    def test_shared_merchant_high(self):
        items = synthesize_network_evidence(
            network_risk_signals={"merchant_shared": True},
            accounts_seen_on_device=[],
            accounts_seen_at_merchant=["A"] * 12,
            account_history_count=10,
            related_transaction_count=5,
        )

        merchant = [
            i for i in items
            if i.title == "Merchant concentration across accounts"
        ]

        assert len(merchant) == 1
        assert merchant[0].severity == "HIGH"

    def test_no_prior_history(self):
        items = synthesize_network_evidence(
            network_risk_signals={},
            accounts_seen_on_device=[],
            accounts_seen_at_merchant=[],
            account_history_count=0,
            related_transaction_count=0,
        )

        titles = [i.title for i in items]

        assert "No prior account history" in titles

    def test_empty_signals(self):
        items = synthesize_network_evidence(
            network_risk_signals=None,
            accounts_seen_on_device=[],
            accounts_seen_at_merchant=[],
            account_history_count=10,
            related_transaction_count=5,
        )

        assert items == []

    def test_high_related_transactions(self):
        items = synthesize_network_evidence(
            network_risk_signals={},
            accounts_seen_on_device=[],
            accounts_seen_at_merchant=[],
            account_history_count=10,
            related_transaction_count=10,
        )

        titles = [i.title for i in items]

        assert "High related transaction volume" in titles


# ============================================================
# CLUSTER EVIDENCE
# ============================================================


class TestClusterEvidence:
    def test_shared_device_signal(self):
        items = synthesize_cluster_evidence(
            cluster_signals=[
                {
                    "type": "SHARED_DEVICE",
                    "severity": "HIGH",
                    "value": 3,
                    "evidence": "3 accounts use device D1",
                }
            ],
            cluster_evidence=[],
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=55.0,
            cluster_accounts=["A", "B", "C"],
            cluster_devices=["D1"],
            cluster_merchants=["M1"],
            cluster_transactions=["T1", "T2", "T3"],
        )

        assert len(items) >= 1

        shared = [
            i for i in items
            if i.title == "Coordinated device usage"
        ]

        assert len(shared) == 1
        assert shared[0].severity == "HIGH"
        assert shared[0].category == "CLUSTER"

    def test_temporal_burst_signal(self):
        items = synthesize_cluster_evidence(
            cluster_signals=[
                {
                    "type": "TEMPORAL_BURST",
                    "severity": "HIGH",
                    "value": 1,
                    "evidence": "1 burst detected",
                }
            ],
            cluster_evidence=[],
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=45.0,
            cluster_accounts=["A", "B"],
            cluster_devices=["D1"],
            cluster_merchants=["M1"],
            cluster_transactions=["T1", "T2", "T3"],
        )

        burst = [
            i for i in items
            if i.title == "Temporal transaction burst"
        ]

        assert len(burst) == 1

    def test_coordinated_network_detection(self):
        items = synthesize_cluster_evidence(
            cluster_signals=[
                {
                    "type": "MULTI_ACCOUNT_CONNECTION",
                    "severity": "MEDIUM",
                    "value": 3,
                    "evidence": "3 connected accounts",
                }
            ],
            cluster_evidence=[],
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=50.0,
            cluster_accounts=["A", "B", "C"],
            cluster_devices=["D1"],
            cluster_merchants=["M1"],
            cluster_transactions=["T1", "T2", "T3"],
        )

        convergence = [
            i for i in items
            if i.title == "Coordinated network detected"
        ]

        assert len(convergence) == 1
        assert convergence[0].severity == "HIGH"

    def test_empty_signals(self):
        items = synthesize_cluster_evidence(
            cluster_signals=[],
            cluster_evidence=[],
            cluster_type="CONNECTED_ACTIVITY",
            cluster_risk_score=10.0,
            cluster_accounts=["A"],
            cluster_devices=["D1"],
            cluster_merchants=["M1"],
            cluster_transactions=["T1"],
        )

        assert items == []

    def test_none_signals(self):
        items = synthesize_cluster_evidence(
            cluster_signals=None,
            cluster_evidence=None,
            cluster_type=None,
            cluster_risk_score=None,
            cluster_accounts=None,
            cluster_devices=None,
            cluster_merchants=None,
            cluster_transactions=None,
        )

        assert items == []


# ============================================================
# CONVERGENCE EVIDENCE
# ============================================================


class TestConvergenceEvidence:
    def test_convergence_with_network_and_cluster(self):
        network_items = [
            EvidenceItem(
                title="Shared device",
                severity="HIGH",
                category="NETWORK",
                explanation="test",
                investigative_relevance="test",
            )
        ]

        cluster_items = [
            EvidenceItem(
                title="Cluster signal",
                severity="HIGH",
                category="CLUSTER",
                explanation="test",
                investigative_relevance="test",
            )
        ]

        items = synthesize_convergence_evidence(
            network_items=network_items,
            cluster_items=cluster_items,
            risk_score=60.0,
            model_probability=0.5,
            network_score=5.0,
        )

        assert len(items) == 1
        assert items[0].category == "CONVERGENCE"

    def test_no_convergence_with_single_signal(self):
        network_items = [
            EvidenceItem(
                title="Shared device",
                severity="HIGH",
                category="NETWORK",
                explanation="test",
                investigative_relevance="test",
            )
        ]

        items = synthesize_convergence_evidence(
            network_items=network_items,
            cluster_items=[],
            risk_score=30.0,
            model_probability=0.3,
            network_score=3.0,
        )

        assert items == []


# ============================================================
# FULL SYNTHESIS
# ============================================================


class TestFullSynthesis:
    def test_full_evidence_bundle(self):
        items = build_coordinated_evidence(
            network_risk_signals={
                "device_shared": True,
                "new_device_for_account": True,
            },
            accounts_seen_on_device=["A", "B", "C"],
            accounts_seen_at_merchant=["X", "Y"],
            account_history_count=5,
            related_transaction_count=8,
            cluster_signals=[
                {
                    "type": "SHARED_DEVICE",
                    "severity": "HIGH",
                    "value": 3,
                    "evidence": "3 accounts use device D1",
                }
            ],
            cluster_evidence=["3 accounts connected"],
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=55.0,
            cluster_accounts=["A", "B", "C"],
            cluster_devices=["D1"],
            cluster_merchants=["M1"],
            cluster_transactions=["T1", "T2", "T3"],
            risk_score=75.0,
            model_probability=0.75,
            network_score=8.0,
        )

        assert len(items) > 0

        categories = {i.category for i in items}

        assert "CONVERGENCE" in categories
        assert "CLUSTER" in categories
        assert "NETWORK" in categories


# ============================================================
# EVIDENCE TO DICT
# ============================================================


class TestEvidenceToDict:
    def test_roundtrip(self):
        item = EvidenceItem(
            title="Test",
            severity="HIGH",
            category="NETWORK",
            explanation="Explanation",
            investigative_relevance="Relevance",
            supporting_entities=["entity:1"],
            supporting_transactions=["T1"],
            observed_value="value",
        )

        d = evidence_to_dict(item)

        assert d["title"] == "Test"
        assert d["severity"] == "HIGH"
        assert d["category"] == "NETWORK"
        assert d["supporting_entities"] == ["entity:1"]
        assert d["supporting_transactions"] == ["T1"]
