from __future__ import annotations

from enum import Enum


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"


ALLOWED_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.OPEN: {
        CaseStatus.INVESTIGATING,
    },
    CaseStatus.INVESTIGATING: {
        CaseStatus.RESOLVED,
    },
    CaseStatus.RESOLVED: set(),
}


def can_transition(
    current: CaseStatus | str,
    target: CaseStatus | str,
) -> bool:
    """
    Return whether a case is allowed to move between states.
    """

    current_status = CaseStatus(current)
    target_status = CaseStatus(target)

    return target_status in ALLOWED_TRANSITIONS[
        current_status
    ]


def transition_case(
    current: CaseStatus | str,
    target: CaseStatus | str,
) -> CaseStatus:
    """
    Perform a validated case-state transition.

    Invalid transitions fail closed.
    """

    current_status = CaseStatus(current)
    target_status = CaseStatus(target)

    if not can_transition(
        current_status,
        target_status,
    ):
        raise ValueError(
            f"Invalid case transition: "
            f"{current_status.value} -> "
            f"{target_status.value}"
        )

    return target_status