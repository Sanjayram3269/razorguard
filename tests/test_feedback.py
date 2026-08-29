from __future__ import annotations

import pytest
import pandas as pd

from razorguard.feedback.outcomes import (
    CaseOutcome,
    InvestigationOutcome,
    OutcomeConfidence,
    is_actionable_feedback,
    is_negative_feedback,
    is_positive_feedback,
)

from razorguard.feedback.metrics import (
    compute_outcome_metrics,
    decision_band_metrics,
    evaluate_decisions,
)

def test_compute_outcome_metrics():
    outcomes = [
        {
            "case_id": "C1",
            "transaction_id": "T1",
            "outcome": "confirmed_fraud",
        },
        {
            "case_id": "C2",
            "transaction_id": "T2",
            "outcome": "legitimate",
        },
        {
            "case_id": "C3",
            "transaction_id": "T3",
            "outcome": "dismissed",
        },
        {
            "case_id": "C4",
            "transaction_id": "T4",
            "outcome": "escalated",
        },
    ]

    metrics = compute_outcome_metrics(outcomes)

    assert metrics["total_outcomes"] == 4
    assert metrics["actionable_outcomes"] == 3
    assert metrics["confirmed_fraud"] == 1
    assert metrics["legitimate"] == 1
    assert metrics["dismissed"] == 1
    assert metrics["escalated"] == 1
    assert metrics["confirmation_rate"] == pytest.approx(1 / 3)


def test_compute_outcome_metrics_empty():
    metrics = compute_outcome_metrics([])

    assert metrics["total_outcomes"] == 0
    assert metrics["actionable_outcomes"] == 0
    assert metrics["confirmation_rate"] == 0.0


def test_evaluate_decisions():
    cases = pd.DataFrame(
        {
            "case_id": ["C1", "C2", "C3", "C4"],
            "decision": [
                "BLOCK",
                "REVIEW",
                "REVIEW",
                "ALLOW",
            ],
        }
    )

    outcomes = [
        {
            "case_id": "C1",
            "transaction_id": "T1",
            "outcome": "confirmed_fraud",
        },
        {
            "case_id": "C2",
            "transaction_id": "T2",
            "outcome": "legitimate",
        },
        {
            "case_id": "C3",
            "transaction_id": "T3",
            "outcome": "confirmed_fraud",
        },
        {
            "case_id": "C4",
            "transaction_id": "T4",
            "outcome": "legitimate",
        },
    ]

    metrics = evaluate_decisions(
        cases,
        outcomes,
    )

    assert metrics["evaluated_cases"] == 4
    assert metrics["confirmed_fraud"] == 2
    assert metrics["non_fraud"] == 2
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 0


def test_evaluate_decisions_ignores_non_definitive_outcomes():
    cases = pd.DataFrame(
        {
            "case_id": ["C1"],
            "decision": ["REVIEW"],
        }
    )

    outcomes = [
        {
            "case_id": "C1",
            "transaction_id": "T1",
            "outcome": "insufficient_evidence",
        }
    ]

    metrics = evaluate_decisions(
        cases,
        outcomes,
    )

    assert metrics["evaluated_cases"] == 0


def test_decision_band_metrics():
    cases = pd.DataFrame(
        {
            "case_id": ["C1", "C2", "C3"],
            "decision": [
                "BLOCK",
                "REVIEW",
                "REVIEW",
            ],
        }
    )

    outcomes = [
        {
            "case_id": "C1",
            "transaction_id": "T1",
            "outcome": "confirmed_fraud",
        },
        {
            "case_id": "C2",
            "transaction_id": "T2",
            "outcome": "legitimate",
        },
        {
            "case_id": "C3",
            "transaction_id": "T3",
            "outcome": "confirmed_fraud",
        },
    ]

    result = decision_band_metrics(
        cases,
        outcomes,
    )

    assert set(result["decision"]) == {
        "BLOCK",
        "REVIEW",
    }

    review = result[
        result["decision"] == "REVIEW"
    ].iloc[0]

    assert review["evaluated_cases"] == 2
    assert review["confirmed_fraud"] == 1
    assert review["non_fraud"] == 1
    assert review["confirmation_rate"] == 0.5


def test_investigation_outcome_serializes_cleanly():
    outcome = InvestigationOutcome(
        case_id="C001",
        transaction_id="T001",
        outcome=CaseOutcome.CONFIRMED_FRAUD,
        confidence=OutcomeConfidence.HIGH,
        investigator="analyst-01",
        notes="Multiple independent fraud indicators confirmed.",
        created_at="2026-08-29T10:00:00+00:00",
    )

    data = outcome.to_dict()

    assert data["case_id"] == "C001"
    assert data["transaction_id"] == "T001"
    assert data["outcome"] == "confirmed_fraud"
    assert data["confidence"] == "high"
    assert data["investigator"] == "analyst-01"


def test_created_at_is_generated_when_missing():
    outcome = InvestigationOutcome(
        case_id="C001",
        transaction_id="T001",
        outcome=CaseOutcome.LEGITIMATE,
        confidence=OutcomeConfidence.MEDIUM,
        investigator="analyst-01",
    )

    assert outcome.created_at
    assert "T" in outcome.created_at


def test_empty_case_id_is_rejected():
    with pytest.raises(ValueError):
        InvestigationOutcome(
            case_id="",
            transaction_id="T001",
            outcome=CaseOutcome.LEGITIMATE,
            confidence=OutcomeConfidence.LOW,
            investigator="analyst-01",
        )


def test_empty_transaction_id_is_rejected():
    with pytest.raises(ValueError):
        InvestigationOutcome(
            case_id="C001",
            transaction_id="",
            outcome=CaseOutcome.LEGITIMATE,
            confidence=OutcomeConfidence.LOW,
            investigator="analyst-01",
        )


def test_empty_investigator_is_rejected():
    with pytest.raises(ValueError):
        InvestigationOutcome(
            case_id="C001",
            transaction_id="T001",
            outcome=CaseOutcome.LEGITIMATE,
            confidence=OutcomeConfidence.LOW,
            investigator="",
        )


def test_confirmed_fraud_requires_notes():
    with pytest.raises(ValueError):
        InvestigationOutcome(
            case_id="C001",
            transaction_id="T001",
            outcome=CaseOutcome.CONFIRMED_FRAUD,
            confidence=OutcomeConfidence.HIGH,
            investigator="analyst-01",
        )


def test_confirmed_fraud_is_positive_feedback():
    assert is_positive_feedback(
        CaseOutcome.CONFIRMED_FRAUD
    )


def test_legitimate_is_negative_feedback():
    assert is_negative_feedback(
        CaseOutcome.LEGITIMATE
    )


def test_dismissed_is_negative_feedback():
    assert is_negative_feedback(
        CaseOutcome.DISMISSED
    )


def test_escalated_is_not_evaluation_label():
    assert not is_actionable_feedback(
        CaseOutcome.ESCALATED
    )


def test_insufficient_evidence_is_not_evaluation_label():
    assert not is_actionable_feedback(
        CaseOutcome.INSUFFICIENT_EVIDENCE
    )


def test_confirmed_fraud_is_actionable():
    assert is_actionable_feedback(
        CaseOutcome.CONFIRMED_FRAUD
    )


def test_legitimate_is_actionable():
    assert is_actionable_feedback(
        CaseOutcome.LEGITIMATE
    )