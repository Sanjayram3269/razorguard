from __future__ import annotations


def policy_decision(
    risk_score: float,
) -> str:
    """
    Deterministic policy gate.

    The model never directly executes a financial action.
    """

    score = float(risk_score)

    if score >= 85:
        return "BLOCK"

    if score >= 55:
        return "REVIEW"

    return "ALLOW"