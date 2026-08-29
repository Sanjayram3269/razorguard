from __future__ import annotations

import pytest

from razorguard.investigation.actions import (
    InvestigatorAction,
    action_resolution,
    validate_action,
)
from razorguard.investigation.audit import (
    create_audit_event,
)
from razorguard.investigation.lifecycle import (
    CaseStatus,
    can_transition,
    transition_case,
)


def test_initial_case_is_open():
    assert CaseStatus.OPEN.value == "OPEN"


def test_open_can_become_investigating():
    assert can_transition(
        CaseStatus.OPEN,
        CaseStatus.INVESTIGATING,
    )


def test_investigating_can_become_resolved():
    assert can_transition(
        CaseStatus.INVESTIGATING,
        CaseStatus.RESOLVED,
    )


def test_open_cannot_skip_to_resolved():
    assert not can_transition(
        CaseStatus.OPEN,
        CaseStatus.RESOLVED,
    )


def test_resolved_is_terminal():
    assert not can_transition(
        CaseStatus.RESOLVED,
        CaseStatus.OPEN,
    )

    assert not can_transition(
        CaseStatus.RESOLVED,
        CaseStatus.INVESTIGATING,
    )


def test_valid_transition_returns_target_state():
    result = transition_case(
        CaseStatus.OPEN,
        CaseStatus.INVESTIGATING,
    )

    assert result == CaseStatus.INVESTIGATING


def test_invalid_transition_fails_closed():
    with pytest.raises(ValueError):
        transition_case(
            CaseStatus.OPEN,
            CaseStatus.RESOLVED,
        )


def test_confirm_fraud_action():
    assert validate_action(
        "CONFIRM_FRAUD"
    ) == InvestigatorAction.CONFIRM_FRAUD

    assert action_resolution(
        InvestigatorAction.CONFIRM_FRAUD
    ) == "FRAUD_CONFIRMED"


def test_legitimate_action():
    assert action_resolution(
        "MARK_LEGITIMATE"
    ) == "LEGITIMATE_CONFIRMED"


def test_escalation_action():
    assert action_resolution(
        "ESCALATE"
    ) == "ESCALATED"


def test_invalid_action_fails():
    with pytest.raises(ValueError):
        validate_action("INVALID_ACTION")


def test_audit_event_is_deterministic_when_timestamp_supplied():
    event = create_audit_event(
        case_id="CASE-000001",
        actor="investigator-01",
        action="START_INVESTIGATION",
        previous_state="OPEN",
        new_state="INVESTIGATING",
        reason="Manual investigation started",
        timestamp="2026-08-29T10:00:00+00:00",
    )

    assert event.case_id == "CASE-000001"
    assert event.actor == "investigator-01"
    assert event.previous_state == "OPEN"
    assert event.new_state == "INVESTIGATING"
    assert event.timestamp == "2026-08-29T10:00:00+00:00"


def test_audit_event_serializes():
    event = create_audit_event(
        case_id="CASE-000002",
        actor="system",
        action="ESCALATE",
        previous_state="INVESTIGATING",
        new_state="INVESTIGATING",
        reason="High-risk case requires senior review",
        timestamp="2026-08-29T10:00:00+00:00",
    )

    payload = event.to_dict()

    assert payload["case_id"] == "CASE-000002"
    assert payload["action"] == "ESCALATE"
    assert payload["new_state"] == "INVESTIGATING"


def test_audit_requires_case_id():
    with pytest.raises(ValueError):
        create_audit_event(
            case_id="",
            actor="system",
            action="TEST",
            previous_state="OPEN",
            new_state="INVESTIGATING",
            reason="test",
        )


def test_audit_requires_actor():
    with pytest.raises(ValueError):
        create_audit_event(
            case_id="CASE-1",
            actor="",
            action="TEST",
            previous_state="OPEN",
            new_state="INVESTIGATING",
            reason="test",
        )


def test_audit_requires_reason():
    with pytest.raises(ValueError):
        create_audit_event(
            case_id="CASE-1",
            actor="system",
            action="TEST",
            previous_state="OPEN",
            new_state="INVESTIGATING",
            reason="",
        )