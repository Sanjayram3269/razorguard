from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class CaseOutcome(str, Enum):
    """Canonical investigator disposition for a risk case."""

    CONFIRMED_FRAUD = "confirmed_fraud"
    LEGITIMATE = "legitimate"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class OutcomeConfidence(str, Enum):
    """Confidence attached to the investigator's disposition."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class InvestigationOutcome:
    """
    Immutable investigator outcome attached to a RiskCase.

    The outcome represents human investigation feedback and is kept
    separate from the model's original prediction.
    """

    case_id: str
    transaction_id: str
    outcome: CaseOutcome
    confidence: OutcomeConfidence
    investigator: str
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")

        if not self.transaction_id:
            raise ValueError("transaction_id must not be empty")

        if not self.investigator:
            raise ValueError("investigator must not be empty")

        if not self.notes.strip() and self.outcome == CaseOutcome.CONFIRMED_FRAUD:
            raise ValueError(
                "confirmed_fraud outcomes require investigator notes"
            )

        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe representation."""

        data = asdict(self)

        data["outcome"] = self.outcome.value
        data["confidence"] = self.confidence.value

        return data


def is_positive_feedback(outcome: CaseOutcome) -> bool:
    """
    Return whether the investigator confirmed fraudulent behavior.
    """

    return outcome == CaseOutcome.CONFIRMED_FRAUD


def is_negative_feedback(outcome: CaseOutcome) -> bool:
    """
    Return whether the investigation established the transaction
    as non-fraudulent.
    """

    return outcome in {
        CaseOutcome.LEGITIMATE,
        CaseOutcome.DISMISSED,
    }


def is_actionable_feedback(outcome: CaseOutcome) -> bool:
    """
    Return whether the outcome provides a definitive evaluation signal.

    Escalated and insufficient-evidence cases remain useful operationally,
    but should not be treated as confirmed labels during model evaluation.
    """

    return outcome in {
        CaseOutcome.CONFIRMED_FRAUD,
        CaseOutcome.LEGITIMATE,
        CaseOutcome.DISMISSED,
    }