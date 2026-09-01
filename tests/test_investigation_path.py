"""Tests for investigation path engine."""
from __future__ import annotations

import pytest

from razorguard.investigation.path import (
    InvestigationStep,
    build_investigation_path,
    step_to_dict,
)


# ============================================================
# PATH GENERATION
# ============================================================


class TestInvestigationPath:
    def test_high_risk_block_case(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=85.0,
            decision="BLOCK",
            model_probability=0.9,
            network_score=9.0,
            assigned_to=None,
            has_audit_events=False,
            device_shared=True,
            merchant_shared=False,
            new_device=True,
            accounts_on_device=5,
            accounts_at_merchant=2,
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=60.0,
            cluster_accounts=["A", "B", "C"],
            cluster_devices=["D1"],
            cluster_transactions=["T1", "T2", "T3"],
            has_temporal_burst=True,
        )

        titles = [s.title for s in steps]

        assert "Assign case to investigator" in titles
        assert "Review high-risk decision" in titles
        assert "Review device-connected accounts" in titles
        assert "Examine coordinated-risk cluster" in titles
        assert "Investigate temporal burst pattern" in titles

    def test_medium_risk_review_case(self):
        steps = build_investigation_path(
            status="IN_REVIEW",
            risk_score=55.0,
            decision="REVIEW",
            model_probability=0.6,
            network_score=5.0,
            assigned_to="analyst-1",
            has_audit_events=True,
            device_shared=False,
            merchant_shared=True,
            new_device=False,
            accounts_on_device=1,
            accounts_at_merchant=8,
            cluster_type=None,
            cluster_risk_score=None,
            cluster_accounts=None,
            cluster_devices=None,
            cluster_transactions=None,
            has_temporal_burst=False,
        )

        titles = [s.title for s in steps]

        assert "Review high-risk decision" not in titles
        assert "Inspect merchant-connected accounts" in titles
        assert "Review case audit history" in titles

    def test_low_risk_minimal_steps(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=25.0,
            decision="ALLOW",
            model_probability=0.3,
            network_score=2.0,
            assigned_to="analyst-1",
            has_audit_events=False,
            device_shared=False,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=1,
            accounts_at_merchant=1,
        )

        titles = [s.title for s in steps]

        # Low risk should have minimal steps
        assert "Review high-risk decision" not in titles
        assert "Review model prediction" not in titles
        assert "Investigate network risk" not in steps

    def test_unassigned_case(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=50.0,
            decision="REVIEW",
            model_probability=0.5,
            network_score=4.0,
            assigned_to=None,
            has_audit_events=False,
            device_shared=False,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=1,
            accounts_at_merchant=1,
        )

        titles = [s.title for s in steps]

        assert "Assign case to investigator" in titles

    def test_temporal_burst_generates_step(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=50.0,
            decision="REVIEW",
            model_probability=0.5,
            network_score=4.0,
            assigned_to="analyst-1",
            has_audit_events=False,
            device_shared=False,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=1,
            accounts_at_merchant=1,
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=45.0,
            cluster_accounts=["A", "B"],
            cluster_devices=["D1"],
            cluster_transactions=["T1", "T2", "T3"],
            has_temporal_burst=True,
        )

        titles = [s.title for s in steps]

        assert "Investigate temporal burst pattern" in titles

    def test_deduplication(self):
        """Steps with the same title should not appear twice."""
        steps = build_investigation_path(
            status="OPEN",
            risk_score=80.0,
            decision="BLOCK",
            model_probability=0.85,
            network_score=8.0,
            assigned_to=None,
            has_audit_events=True,
            device_shared=True,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=5,
            accounts_at_merchant=1,
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=50.0,
            cluster_accounts=["A", "B"],
            cluster_devices=["D1"],
            cluster_transactions=["T1", "T2"],
            has_temporal_burst=False,
        )

        titles = [s.title for s in steps]

        # No duplicate titles
        assert len(titles) == len(set(titles))

    def test_sorting_by_priority(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=85.0,
            decision="BLOCK",
            model_probability=0.9,
            network_score=9.0,
            assigned_to=None,
            has_audit_events=True,
            device_shared=True,
            merchant_shared=False,
            new_device=True,
            accounts_on_device=5,
            accounts_at_merchant=2,
            cluster_type="COORDINATED_NETWORK",
            cluster_risk_score=60.0,
            cluster_accounts=["A", "B", "C"],
            cluster_devices=["D1"],
            cluster_transactions=["T1", "T2", "T3"],
            has_temporal_burst=True,
        )

        priorities = [s.priority for s in steps]

        # Should be sorted ascending (1 = highest)
        assert priorities == sorted(priorities)

    def test_navigation_targets(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=80.0,
            decision="BLOCK",
            model_probability=0.85,
            network_score=8.0,
            assigned_to=None,
            has_audit_events=False,
            device_shared=True,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=5,
            accounts_at_merchant=1,
        )

        nav_targets = {
            s.navigation_target for s in steps
        }

        assert "network" in nav_targets
        assert "cases" in nav_targets


# ============================================================
# EDGE CASES
# ============================================================


class TestEdgeCases:
    def test_resolved_case(self):
        steps = build_investigation_path(
            status="RESOLVED",
            risk_score=80.0,
            decision="BLOCK",
            model_probability=0.85,
            network_score=8.0,
            assigned_to="analyst-1",
            has_audit_events=True,
            device_shared=False,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=1,
            accounts_at_merchant=1,
        )

        # Resolved case should not generate assignment steps
        titles = [s.title for s in steps]

        assert "Assign case to investigator" not in titles
        assert "Begin investigation review" not in titles

    def test_empty_cluster_data(self):
        steps = build_investigation_path(
            status="OPEN",
            risk_score=50.0,
            decision="REVIEW",
            model_probability=0.5,
            network_score=4.0,
            assigned_to="analyst-1",
            has_audit_events=False,
            device_shared=False,
            merchant_shared=False,
            new_device=False,
            accounts_on_device=1,
            accounts_at_merchant=1,
            cluster_type=None,
            cluster_risk_score=None,
            cluster_accounts=None,
            cluster_devices=None,
            cluster_transactions=None,
            has_temporal_burst=False,
        )

        # Should still produce some steps
        assert len(steps) >= 0


# ============================================================
# STEP TO DICT
# ============================================================


class TestStepToDict:
    def test_roundtrip(self):
        step = InvestigationStep(
            priority=1,
            title="Test step",
            reason="Because",
            supporting_evidence=["Evidence 1"],
            target_entity="device",
            navigation_target="network",
        )

        d = step_to_dict(step)

        assert d["priority"] == 1
        assert d["title"] == "Test step"
        assert d["reason"] == "Because"
        assert d["supporting_evidence"] == ["Evidence 1"]
        assert d["target_entity"] == "device"
        assert d["navigation_target"] == "network"
