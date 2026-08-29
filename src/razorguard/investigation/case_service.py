from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from razorguard.investigation.actions import (
    InvestigatorAction,
    action_resolution,
)
from razorguard.investigation.audit import (
    AuditEvent,
    create_audit_event,
)
from razorguard.investigation.lifecycle import (
    CaseStatus,
    transition_case,
)
from razorguard.risk.case import RiskCase


@dataclass
class CaseRecord:
    """
    Mutable investigator-facing case record.

    RiskCase remains the immutable output of the risk engine.
    CaseRecord adds operational state around that decision.
    """

    case_id: str
    risk_case: RiskCase
    status: CaseStatus = CaseStatus.OPEN

    investigator: str | None = None
    resolution: str | None = None
    resolution_reason: str | None = None

    audit_events: list[AuditEvent] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the complete case record.
        """

        return {
            "case_id": self.case_id,
            "transaction_id": (
                self.risk_case.transaction_id
            ),
            "risk_score": self.risk_case.risk_score,
            "risk_level": self.risk_case.risk_level,
            "decision": self.risk_case.decision,
            "primary_reason": (
                self.risk_case.primary_reason
            ),
            "evidence": list(
                self.risk_case.evidence
            ),
            "model_probability": (
                self.risk_case.model_probability
            ),
            "network_score": (
                self.risk_case.network_score
            ),
            "status": self.status.value,
            "investigator": self.investigator,
            "resolution": self.resolution,
            "resolution_reason": (
                self.resolution_reason
            ),
            "audit_events": [
                event.to_dict()
                for event in self.audit_events
            ],
        }


def generate_case_id(
    transaction_id: str,
    sequence: int,
) -> str:
    """
    Generate a deterministic case identifier.

    Example:
        CASE-T001-000001
    """

    if not transaction_id:
        raise ValueError(
            "transaction_id must not be empty"
        )

    if sequence < 1:
        raise ValueError(
            "sequence must be >= 1"
        )

    return (
        f"CASE-{transaction_id}-"
        f"{sequence:06d}"
    )


def create_case(
    risk_case: RiskCase,
    sequence: int,
    actor: str = "risk-engine",
) -> CaseRecord:
    """
    Create an investigation case from a RiskCase.

    Every created case starts in OPEN state and receives
    an immutable creation audit event.
    """

    case_id = generate_case_id(
        risk_case.transaction_id,
        sequence,
    )

    record = CaseRecord(
        case_id=case_id,
        risk_case=risk_case,
    )

    event = create_audit_event(
        case_id=case_id,
        actor=actor,
        action="CASE_CREATED",
        previous_state="NONE",
        new_state=CaseStatus.OPEN.value,
        reason="Risk engine created investigation case",
    )

    record.audit_events.append(event)

    return record


def start_investigation(
    case: CaseRecord,
    investigator: str,
    reason: str = "Investigation started",
) -> CaseRecord:
    """
    Move an OPEN case into INVESTIGATING state.
    """

    if not investigator:
        raise ValueError(
            "investigator must not be empty"
        )

    previous = case.status

    case.status = transition_case(
        previous,
        CaseStatus.INVESTIGATING,
    )

    case.investigator = investigator

    event = create_audit_event(
        case_id=case.case_id,
        actor=investigator,
        action="START_INVESTIGATION",
        previous_state=previous.value,
        new_state=case.status.value,
        reason=reason,
    )

    case.audit_events.append(event)

    return case


def apply_action(
    case: CaseRecord,
    action: InvestigatorAction | str,
    actor: str,
    reason: str,
) -> CaseRecord:
    """
    Apply an investigator action to a case.

    Terminal actions resolve the case.
    ESCALATE keeps the case under investigation.
    """

    if not actor:
        raise ValueError(
            "actor must not be empty"
        )

    if not reason:
        raise ValueError(
            "reason must not be empty"
        )

    normalized_action = InvestigatorAction(
        action
    )

    # Cases must be actively investigated before
    # an investigator can resolve or escalate them.
    if case.status != CaseStatus.INVESTIGATING:
        raise ValueError(
            "investigator actions require "
            "an INVESTIGATING case"
        )

    previous = case.status

    resolution = action_resolution(
        normalized_action
    )

    if normalized_action in {
        InvestigatorAction.CONFIRM_FRAUD,
        InvestigatorAction.MARK_LEGITIMATE,
    }:
        case.status = transition_case(
            previous,
            CaseStatus.RESOLVED,
        )

        case.resolution = resolution
        case.resolution_reason = reason

    else:
        # Escalation does not resolve the case.
        case.status = CaseStatus.INVESTIGATING

        case.resolution = None
        case.resolution_reason = None

    event = create_audit_event(
        case_id=case.case_id,
        actor=actor,
        action=normalized_action.value,
        previous_state=previous.value,
        new_state=case.status.value,
        reason=reason,
    )

    case.audit_events.append(event)

    return case