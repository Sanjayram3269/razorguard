from __future__ import annotations

import pytest

from razorguard.investigation.store import CaseStore
from razorguard.risk.case import RiskCase


def make_case(
    transaction_id: str = "T-LIFE",
) -> RiskCase:
    return RiskCase(
        transaction_id=transaction_id,
        risk_score=0.86,
        risk_level="HIGH",
        decision="REVIEW",
        primary_reason="test risk",
        evidence=[
            "test evidence",
        ],
        model_probability=0.91,
        network_score=0.72,
    )


def make_store(tmp_path) -> CaseStore:
    return CaseStore(
        tmp_path / "cases.parquet"
    )


def test_open_to_in_review(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    case = store.update_status(
        "CASE-T-LIFE",
        "IN_REVIEW",
    )

    assert case["status"] == "IN_REVIEW"


def test_in_review_to_escalated(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    store.update_status(
        "CASE-T-LIFE",
        "IN_REVIEW",
    )

    case = store.update_status(
        "CASE-T-LIFE",
        "ESCALATED",
    )

    assert case["status"] == "ESCALATED"


def test_escalated_can_return_to_review(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    store.update_status(
        "CASE-T-LIFE",
        "ESCALATED",
    )

    case = store.update_status(
        "CASE-T-LIFE",
        "IN_REVIEW",
    )

    assert case["status"] == "IN_REVIEW"


def test_terminal_case_cannot_reopen(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    store.update_status(
        "CASE-T-LIFE",
        "RESOLVED",
    )

    with pytest.raises(
        ValueError,
        match="already terminal",
    ):
        store.update_status(
            "CASE-T-LIFE",
            "IN_REVIEW",
        )


def test_same_status_is_rejected(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    with pytest.raises(
        ValueError,
        match="already in OPEN",
    ):
        store.update_status(
            "CASE-T-LIFE",
            "OPEN",
        )


def test_invalid_transition_is_rejected(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    store.update_status(
        "CASE-T-LIFE",
        "IN_REVIEW",
    )

    with pytest.raises(
        ValueError,
        match="invalid transition",
    ):
        store.update_status(
            "CASE-T-LIFE",
            "OPEN",
        )


def test_lifecycle_transitions_are_audited(tmp_path):
    store = make_store(tmp_path)

    store.create(
        make_case()
    )

    store.update_status(
        "CASE-T-LIFE",
        "IN_REVIEW",
        actor="investigator",
        details="Started investigation",
    )

    store.update_status(
        "CASE-T-LIFE",
        "ESCALATED",
        actor="investigator",
        details="Escalating for review",
    )

    audit = store.audit(
        "CASE-T-LIFE"
    )

    assert len(audit) == 3

    assert (
        audit.iloc[0]["action"]
        == "CASE_CREATED"
    )

    assert (
        audit.iloc[1]["from_status"]
        == "OPEN"
    )
    assert (
        audit.iloc[1]["to_status"]
        == "IN_REVIEW"
    )

    assert (
        audit.iloc[2]["from_status"]
        == "IN_REVIEW"
    )
    assert (
        audit.iloc[2]["to_status"]
        == "ESCALATED"
    )

    assert (
        audit.iloc[1]["actor"]
        == "investigator"
    )

    assert (
        audit.iloc[2]["details"]
        == "Escalating for review"
    )