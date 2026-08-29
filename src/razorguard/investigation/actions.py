from __future__ import annotations

from enum import Enum


class InvestigatorAction(str, Enum):
    CONFIRM_FRAUD = "CONFIRM_FRAUD"
    MARK_LEGITIMATE = "MARK_LEGITIMATE"
    ESCALATE = "ESCALATE"


TERMINAL_ACTIONS = {
    InvestigatorAction.CONFIRM_FRAUD,
    InvestigatorAction.MARK_LEGITIMATE,
}


def validate_action(
    action: InvestigatorAction | str,
) -> InvestigatorAction:
    """
    Validate and normalize an investigator action.
    """

    return InvestigatorAction(action)


def action_resolution(
    action: InvestigatorAction | str,
) -> str:
    """
    Map investigator action to an explicit resolution.
    """

    normalized = validate_action(action)

    if normalized == InvestigatorAction.CONFIRM_FRAUD:
        return "FRAUD_CONFIRMED"

    if normalized == InvestigatorAction.MARK_LEGITIMATE:
        return "LEGITIMATE_CONFIRMED"

    return "ESCALATED"