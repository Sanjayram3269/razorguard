from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    case_id: str
    timestamp: str
    actor: str
    action: str
    previous_state: str
    new_state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_audit_event(
    case_id: str,
    actor: str,
    action: str,
    previous_state: str,
    new_state: str,
    reason: str,
    timestamp: str | None = None,
) -> AuditEvent:
    """
    Create an immutable audit event.

    UTC timestamps are used so events remain comparable across
    investigators and services.
    """

    if not case_id:
        raise ValueError("case_id must not be empty")

    if not actor:
        raise ValueError("actor must not be empty")

    if not reason:
        raise ValueError("reason must not be empty")

    event_timestamp = timestamp or datetime.now(
        timezone.utc
    ).isoformat()

    return AuditEvent(
        case_id=case_id,
        timestamp=event_timestamp,
        actor=actor,
        action=action,
        previous_state=previous_state,
        new_state=new_state,
        reason=reason,
    )