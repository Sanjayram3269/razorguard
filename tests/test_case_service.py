from __future__ import annotations

import pytest

from razorguard.investigation.actions import (
    InvestigatorAction,
)
from razorguard.investigation.case_service import (
    apply_action,
    create_case,
    generate_case_id,
    start_investigation,
)
from razorguard.investigation.lifecycle import (
    CaseStatus,
)
from razorguard.risk.case import RiskCase


def make_risk_case() -> RiskCase:
    return RiskCase(
        transaction_id="T000001",
        risk_score=92.5,
        risk_level="CRITICAL",
        decision="BLOCK",
        primary_reason=(
            "high combined transaction and network risk"
        ),
        evidence=[
            "transaction amount is elevated",
            "elevated network/entity risk",
        ],
        model_probability=0.95,
        network_score=18.0,
    )


def test_generate_case_id():
    assert (
        generate_case_id(
            "T000001",
            1,
        )
        == "CASE-T000001-000001"
    )


def test_generate_case_id_increments_sequence():
    assert (
        generate_case_id(
            "T000001",
            27,
        )
        == "CASE-T000001-000027"
    )


def test_invalid_case_id_inputs():
    with pytest.raises(ValueError):
        generate_case_id("", 1)

    with pytest.raises(ValueError):
        generate_case_id("T1", 0)


def test_create_case_starts_open():
    case = create_case(
        make_risk_case(),
        sequence=1,
    )

    assert case.case_id == "CASE-T000001-000001"
    assert case.status == CaseStatus.OPEN
    assert case.investigator is None
    assert case.resolution is None
    assert len(case.audit_events) == 1
    assert (
        case.audit_events[0].action
        == "CASE_CREATED"
    )


def test_start_investigation():
    case = create_case(
        make_risk_case(),
        sequence=1,
    )

    start_investigation(
        case,
        investigator="investigator-01",
        reason="High-risk case requires review",
    )

    assert case.status == CaseStatus.INVESTIGATING
    assert (
        case.investigator
        == "investigator-01"
    )
    assert len(case.audit_events) == 2

    event = case.audit_events[-1]

    assert (
        event.action
        == "START_INVESTIGATION"
    )
    assert event.previous_state == "OPEN"
    assert (
        event.new_state
        == "INVESTIGATING"
    )


def test_start_investigation_requires_investigator():
    case = create_case(
        make_risk_case(),
        sequence=1,
    )

    with pytest.raises(ValueError):
        start_investigation(
            case,
            investigator="",
        )


def test_cannot_start_investigation_twice():
    case = create_case(
        make_risk_case(),
        sequence=1,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    with pytest.raises(ValueError):
        start_investigation(
            case,
            investigator="investigator-02",
        )


def test_confirm_fraud_resolves_case():
    case = create_case(
        make_risk_case(),
        sequence=1,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    apply_action(
        case,
        InvestigatorAction.CONFIRM_FRAUD,
        actor="investigator-01",
        reason="Confirmed coordinated fraud pattern",
    )

    assert case.status == CaseStatus.RESOLVED
    assert (
        case.resolution
        == "FRAUD_CONFIRMED"
    )
    assert (
        case.resolution_reason
        == "Confirmed coordinated fraud pattern"
    )

    event = case.audit_events[-1]

    assert event.action == "CONFIRM_FRAUD"
    assert (
        event.previous_state
        == "INVESTIGATING"
    )
    assert (
        event.new_state
        == "RESOLVED"
    )


def test_mark_legitimate_resolves_case():
    case = create_case(
        make_risk_case(),
        sequence=2,
    )

    start_investigation(
        case,
        investigator="investigator-02",
    )

    apply_action(
        case,
        InvestigatorAction.MARK_LEGITIMATE,
        actor="investigator-02",
        reason="Customer verified transaction",
    )

    assert case.status == CaseStatus.RESOLVED
    assert (
        case.resolution
        == "LEGITIMATE_CONFIRMED"
    )


def test_escalation_keeps_case_open_for_investigation():
    case = create_case(
        make_risk_case(),
        sequence=3,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    apply_action(
        case,
        InvestigatorAction.ESCALATE,
        actor="investigator-01",
        reason="Requires senior risk review",
    )

    assert (
        case.status
        == CaseStatus.INVESTIGATING
    )
    assert case.resolution is None
    assert case.resolution_reason is None

    event = case.audit_events[-1]

    assert event.action == "ESCALATE"
    assert (
        event.previous_state
        == "INVESTIGATING"
    )
    assert (
        event.new_state
        == "INVESTIGATING"
    )


def test_action_requires_investigating_state():
    case = create_case(
        make_risk_case(),
        sequence=4,
    )

    with pytest.raises(ValueError):
        apply_action(
            case,
            InvestigatorAction.CONFIRM_FRAUD,
            actor="investigator-01",
            reason="test",
        )


def test_resolved_case_cannot_be_modified():
    case = create_case(
        make_risk_case(),
        sequence=5,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    apply_action(
        case,
        InvestigatorAction.CONFIRM_FRAUD,
        actor="investigator-01",
        reason="Fraud confirmed",
    )

    with pytest.raises(ValueError):
        apply_action(
            case,
            InvestigatorAction.MARK_LEGITIMATE,
            actor="investigator-01",
            reason="Attempted reversal",
        )


def test_action_requires_actor():
    case = create_case(
        make_risk_case(),
        sequence=6,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    with pytest.raises(ValueError):
        apply_action(
            case,
            InvestigatorAction.ESCALATE,
            actor="",
            reason="Escalation",
        )


def test_action_requires_reason():
    case = create_case(
        make_risk_case(),
        sequence=7,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    with pytest.raises(ValueError):
        apply_action(
            case,
            InvestigatorAction.ESCALATE,
            actor="investigator-01",
            reason="",
        )


def test_case_serialization():
    case = create_case(
        make_risk_case(),
        sequence=8,
    )

    start_investigation(
        case,
        investigator="investigator-01",
    )

    payload = case.to_dict()

    assert (
        payload["case_id"]
        == "CASE-T000001-000008"
    )

    assert (
        payload["transaction_id"]
        == "T000001"
    )

    assert (
        payload["status"]
        == "INVESTIGATING"
    )

    assert (
        payload["investigator"]
        == "investigator-01"
    )

    assert len(
        payload["audit_events"]
    ) == 2