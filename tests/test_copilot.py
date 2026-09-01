"""Tests for the investigation copilot."""
from __future__ import annotations

import pytest

from razorguard.copilot.context import (
    CopilotEvidenceContext,
    AuditContext,
    CaseContext,
    ClusterContext,
    EvidenceContext,
    InvestigationPathContext,
    NetworkContext,
    build_copilot_context,
    context_to_prompt,
)
from razorguard.copilot.provider import (
    CopilotResponse,
    LLMProvider,
    ProviderUnavailableError,
    _NullProvider,
    get_provider,
    reset_provider,
)
from razorguard.copilot.service import (
    answer_question,
    get_copilot_status,
    _fallback_answer,
    _summarize_case,
    _strongest_evidence,
    _why_flagged,
    _what_next,
    _relationships,
    _explain_cluster,
    _what_unusual,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_case():
    return {
        "case_id": "CASE-TX-001",
        "transaction_id": "TX-001",
        "status": "OPEN",
        "decision": "BLOCK",
        "risk_score": 85.0,
        "risk_level": "CRITICAL",
        "model_probability": 0.92,
        "network_score": 8.5,
        "primary_reason": "high combined transaction and network risk",
        "evidence_text": "transaction amount is elevated | device shared",
        "assigned_to": "analyst-1",
        "investigation_narrative": "Test narrative.",
    }


@pytest.fixture
def sample_evidence():
    return [
        {
            "title": "Device shared across accounts",
            "severity": "HIGH",
            "category": "NETWORK",
            "tier": "PRIMARY",
            "explanation": "Device is shared.",
            "investigative_relevance": "Review accounts.",
        },
        {
            "title": "New merchant",
            "severity": "LOW",
            "category": "NETWORK",
            "tier": "CONTEXTUAL",
            "explanation": "New merchant.",
            "investigative_relevance": "Check history.",
        },
    ]


@pytest.fixture
def sample_network():
    return {
        "account_id": "ACC-001",
        "device_id": "DEV-001",
        "merchant_id": "MER-001",
        "account_history_count": 5,
        "related_transaction_count": 8,
        "accounts_seen_on_device": ["ACC-001", "ACC-002", "ACC-003"],
        "accounts_seen_at_merchant": ["ACC-001", "ACC-002"],
        "network_risk_signals": {
            "device_shared": True,
            "merchant_shared": False,
            "new_device_for_account": True,
            "new_merchant_for_account": False,
        },
    }


@pytest.fixture
def sample_cluster():
    return {
        "cluster_type": "COORDINATED_NETWORK",
        "risk_score": 55.0,
        "accounts": ["ACC-001", "ACC-002", "ACC-003"],
        "devices": ["DEV-001"],
        "merchants": ["MER-001"],
        "transactions": ["T1", "T2", "T3"],
        "signals": [
            {"type": "SHARED_DEVICE", "severity": "HIGH", "value": 3, "evidence": "3 accounts share device"}
        ],
        "evidence": ["3 accounts connected through shared device."],
    }


@pytest.fixture
def sample_steps():
    return [
        {"priority": 1, "title": "Review device accounts", "reason": "Device shared."},
        {"priority": 2, "title": "Check merchant", "reason": "Merchant concentration."},
    ]


@pytest.fixture
def sample_audit():
    return [
        {"action": "CASE_CREATED", "actor": "system", "timestamp": "2026-01-01T00:00:00Z", "details": ""},
    ]


# ============================================================
# CONTEXT BUILDER
# ============================================================


class TestContextBuilder:
    def test_builds_correctly(
        self, sample_case, sample_evidence, sample_network,
        sample_cluster, sample_steps, sample_audit,
    ):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=sample_network,
            cluster_data=sample_cluster,
            investigation_steps=sample_steps,
            audit_events=sample_audit,
            evidence_summary={"PRIMARY": 1, "SUPPORTING": 0, "CONTEXTUAL": 1, "TOTAL": 2},
        )

        assert ctx.case.case_id == "CASE-TX-001"
        assert ctx.case.decision == "BLOCK"
        assert ctx.case.risk_score == 85.0
        assert len(ctx.evidence) == 2
        assert ctx.network is not None
        assert ctx.network.device_shared is True
        assert ctx.cluster is not None
        assert ctx.cluster.cluster_type == "COORDINATED_NETWORK"
        assert len(ctx.investigation_path) == 2
        assert len(ctx.audit_events) == 1

    def test_bounds_evidence(self, sample_case):
        many_evidence = [
            {
                "title": f"Evidence {i}",
                "severity": "LOW",
                "category": "NETWORK",
                "tier": "CONTEXTUAL",
                "explanation": "test",
                "investigative_relevance": "test",
            }
            for i in range(30)
        ]

        ctx = build_copilot_context(
            case=sample_case,
            evidence=many_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        assert len(ctx.evidence) == 15  # bounded

    def test_none_network(self, sample_case, sample_evidence):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        assert ctx.network is None
        assert ctx.cluster is None

    def test_context_to_prompt(self, sample_case, sample_evidence, sample_network):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=sample_network,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        prompt = context_to_prompt(ctx)

        assert "CASE-TX-001" in prompt
        assert "BLOCK" in prompt
        assert "ACC-001" in prompt
        assert "DEV-001" in prompt
        assert "EVIDENCE" in prompt
        assert "NETWORK" in prompt


# ============================================================
# PROVIDER
# ============================================================


class TestProvider:
    def test_null_provider_unavailable(self):
        provider = _NullProvider()
        assert provider.is_available() is False

    def test_null_provider_raises(self):
        provider = _NullProvider()
        with pytest.raises(ProviderUnavailableError):
            provider.generate("system", "user")

    def test_get_provider_returns_something(self):
        reset_provider()
        provider = get_provider()
        assert provider is not None
        assert isinstance(provider.is_available(), bool)


# ============================================================
# SERVICE - FALLBACK ANSWERS
# ============================================================


class TestFallbackAnswers:
    def test_summarize_case(self, sample_case, sample_evidence, sample_network):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=sample_network,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={"PRIMARY": 1},
        )

        response = _summarize_case(ctx)

        assert "CASE-TX-001" in response.answer
        assert "BLOCK" in response.answer
        assert response.grounding == "VERIFIED EVIDENCE"

    def test_strongest_evidence(self, sample_case, sample_evidence):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _strongest_evidence(ctx)

        assert "PRIMARY" in response.answer
        assert "Device shared" in response.answer

    def test_why_flagged(self, sample_case, sample_evidence):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _why_flagged(ctx)

        assert "85.0" in response.answer
        assert "BLOCK" in response.answer

    def test_what_next(self, sample_case, sample_steps):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=None,
            cluster_data=None,
            investigation_steps=sample_steps,
            audit_events=[],
            evidence_summary={},
        )

        response = _what_next(ctx)

        assert "Review device accounts" in response.answer
        assert "P1" in response.answer

    def test_relationships(self, sample_case, sample_network):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=sample_network,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _relationships(ctx)

        assert "ACC-001" in response.answer
        assert "shared" in response.answer.lower()

    def test_explain_cluster(self, sample_case, sample_cluster):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=None,
            cluster_data=sample_cluster,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _explain_cluster(ctx)

        assert "COORDINATED_NETWORK" in response.answer
        assert "3 accounts" in response.answer

    def test_what_unusual(self, sample_case, sample_network):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=sample_network,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _what_unusual(ctx)

        assert "New device" in response.answer
        assert "shared" in response.answer.lower()

    def test_no_evidence_strongest(self, sample_case):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _strongest_evidence(ctx)

        assert "No evidence" in response.answer

    def test_no_steps_what_next(self, sample_case):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _what_next(ctx)

        assert "No investigation steps" in response.answer

    def test_no_network_relationships(self, sample_case):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _relationships(ctx)

        assert "No network intelligence" in response.answer

    def test_no_cluster_explain(self, sample_case):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=[],
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        response = _explain_cluster(ctx)

        assert "No coordinated-risk cluster" in response.answer


# ============================================================
# SERVICE - MAIN ANSWER FUNCTION
# ============================================================


class TestAnswerQuestion:
    def test_fallback_answer(
        self, sample_case, sample_evidence, sample_network,
        sample_cluster, sample_steps, sample_audit,
    ):
        response = answer_question(
            question="Why was this case flagged?",
            case_context=sample_case,
            evidence=sample_evidence,
            network_data=sample_network,
            cluster_data=sample_cluster,
            investigation_steps=sample_steps,
            audit_events=sample_audit,
            evidence_summary={"PRIMARY": 1},
        )

        assert isinstance(response, CopilotResponse)
        assert len(response.answer) > 0
        assert response.grounding in (
            "VERIFIED EVIDENCE",
            "AI INTERPRETATION — verify against RazorGuard evidence",
        )

    def test_summarize_question(
        self, sample_case, sample_evidence,
    ):
        response = answer_question(
            question="Summarize this case",
            case_context=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        assert "CASE-TX-001" in response.answer

    def test_unknown_question_falls_back(
        self, sample_case, sample_evidence,
    ):
        response = answer_question(
            question="What is the meaning of life?",
            case_context=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        # Should fall back to summarize
        assert len(response.answer) > 0


# ============================================================
# STATUS
# ============================================================


class TestStatus:
    def test_get_status(self):
        status = get_copilot_status()

        assert "available" in status
        assert "provider" in status
        assert isinstance(status["available"], bool)


# ============================================================
# SECURITY - NO SECRET LEAKAGE
# ============================================================


class TestSecurity:
    def test_no_api_key_in_context(
        self, sample_case, sample_evidence,
    ):
        ctx = build_copilot_context(
            case=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        prompt = context_to_prompt(ctx)

        assert "OPENAI_API_KEY" not in prompt
        assert "api_key" not in prompt.lower()
        assert "secret" not in prompt.lower()
        assert "password" not in prompt.lower()
        assert "token" not in prompt.lower()

    def test_no_api_key_in_response(
        self, sample_case, sample_evidence,
    ):
        response = answer_question(
            question="Summarize this case",
            case_context=sample_case,
            evidence=sample_evidence,
            network_data=None,
            cluster_data=None,
            investigation_steps=[],
            audit_events=[],
            evidence_summary={},
        )

        assert "OPENAI_API_KEY" not in response.answer
        assert "api_key" not in response.answer.lower()
