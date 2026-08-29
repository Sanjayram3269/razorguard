from __future__ import annotations

import pytest

from razorguard.engine.pipeline import score_transaction


def test_low_risk_transaction_is_allowed():
    transaction = {
        "transaction_id": "T001",
        "amount": 100.0,
    }

    case = score_transaction(
        transaction,
        model_probability=0.05,
        network_score=0.0,
        behavioral_signal=0.0,
    )

    assert case.transaction_id == "T001"
    assert case.decision == "ALLOW"
    assert case.risk_level == "LOW"
    assert case.risk_score < 40


def test_medium_risk_transaction_is_reviewed():
    transaction = {
        "transaction_id": "T002",
        "amount": 900.0,
        "location_mismatch": 1,
        "account_id_prior_count_60m": 3,
    }

    case = score_transaction(
        transaction,
        model_probability=0.65,
        network_score=7.0,
        behavioral_signal=0.70,
    )

    assert case.decision == "REVIEW"
    assert case.risk_score >= 55
    assert case.primary_reason


def test_critical_risk_transaction_is_blocked():
    transaction = {
        "transaction_id": "T003",
        "amount": 5000.0,
        "location_mismatch": 1,
        "is_dormant_return": 1,
        "account_id_prior_count_60m": 8,
        "prior_accounts_per_device": 5,
    }

    case = score_transaction(
        transaction,
        model_probability=1.0,
        network_score=20.0,
        behavioral_signal=1.0,
    )

    assert case.decision == "BLOCK"
    assert case.risk_level == "CRITICAL"
    assert case.risk_score == 100.0


def test_probability_is_clipped():
    transaction = {
        "transaction_id": "T004",
    }

    case = score_transaction(
        transaction,
        model_probability=2.0,
        network_score=-5.0,
        behavioral_signal=3.0,
    )

    assert case.model_probability == 1.0
    assert case.network_score == 0.0
    assert case.risk_score == 70.0


def test_network_risk_produces_evidence():
    transaction = {
        "transaction_id": "T005",
    }

    case = score_transaction(
        transaction,
        model_probability=0.10,
        network_score=9.0,
        behavioral_signal=0.10,
    )

    assert "elevated network/entity risk" in case.evidence


def test_behavioral_risk_produces_evidence():
    transaction = {
        "transaction_id": "T006",
    }

    case = score_transaction(
        transaction,
        model_probability=0.20,
        network_score=1.0,
        behavioral_signal=0.90,
    )

    assert (
        "behavior deviates materially from account history"
        in case.evidence
    )


def test_dormancy_and_velocity_are_explainable():
    transaction = {
        "transaction_id": "T007",
        "is_dormant_return": 1,
        "account_id_prior_count_60m": 5,
    }

    case = score_transaction(
        transaction,
        model_probability=0.30,
        network_score=2.0,
        behavioral_signal=0.20,
    )

    assert (
        "transaction follows an extended dormant period"
        in case.evidence
    )

    assert (
        "elevated account transaction velocity"
        in case.evidence
    )


def test_risk_case_is_serializable():
    transaction = {
        "transaction_id": "T008",
    }

    case = score_transaction(
        transaction,
        model_probability=0.20,
        network_score=2.0,
        behavioral_signal=0.10,
    )

    payload = case.to_dict()

    assert payload["transaction_id"] == "T008"
    assert isinstance(payload["evidence"], list)
    assert isinstance(payload["risk_score"], float)