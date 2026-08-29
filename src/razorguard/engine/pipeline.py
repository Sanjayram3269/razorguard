from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from razorguard.risk.case import RiskCase
from razorguard.risk.fusion import fuse_risk, risk_level
from razorguard.risk.policy import policy_decision

def behavioral_signal(row) -> float:
    """
    Convert observable behavioral deviations into [0, 1].

    This signal is deterministic and independent of the
    supervised model probability.
    """

    def value(key: str, default: float = 0.0) -> float:
        try:
            raw = row.get(key, default)
            if raw is None:
                return default
            return float(raw)
        except (TypeError, ValueError):
            return default

    signals = [
        min(
            value("amount_zscore") / 4.0,
            1.0,
        )
        if value("amount_zscore") > 0
        else 0.0,

        1.0
        if value("is_dormant_return") > 0
        else 0.0,

        min(
            value("account_id_prior_count_60m") / 8.0,
            1.0,
        ),

        1.0
        if value("location_mismatch") > 0
        else 0.0,

        min(
            value("account_velocity_ratio"),
            1.0,
        ),
    ]

    return float(
        min(
            max(
                sum(signals) / len(signals),
                0.0,
            ),
            1.0,
        )
    )

def _clip_probability(value: float) -> float:
    """Keep model probability inside the valid probability domain."""
    return max(0.0, min(float(value), 1.0))


def _clip_signal(value: float) -> float:
    """Keep behavioral signal inside the valid signal domain."""
    return max(0.0, min(float(value), 1.0))


def _transaction_value(
    transaction: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    return transaction.get(key, default)


def build_evidence(
    transaction: Mapping[str, Any],
    model_probability: float,
    network_score: float,
    behavioral_signal: float,
) -> list[str]:
    """
    Build deterministic, human-readable evidence.

    Evidence is intentionally derived from observable signals rather
    than exposing model internals.
    """

    evidence: list[str] = []

    amount = _transaction_value(
        transaction,
        "amount",
    )

    if amount is not None and float(amount) >= 650:
        evidence.append(
            "transaction amount is elevated"
        )

    location_mismatch = _transaction_value(
        transaction,
        "location_mismatch",
        0,
    )

    if bool(location_mismatch):
        evidence.append(
            "IP and shipping countries differ"
        )

    dormant_return = _transaction_value(
        transaction,
        "is_dormant_return",
        0,
    )

    if bool(dormant_return):
        evidence.append(
            "transaction follows an extended dormant period"
        )

    velocity_60m = _transaction_value(
        transaction,
        "account_id_prior_count_60m",
        0,
    )

    if float(velocity_60m) >= 3:
        evidence.append(
            "elevated account transaction velocity"
        )

    unique_merchants = _transaction_value(
        transaction,
        "prior_unique_merchants",
        0,
    )

    if float(unique_merchants) >= 10:
        evidence.append(
            "account has broad prior merchant activity"
        )

    device_accounts = _transaction_value(
        transaction,
        "prior_accounts_per_device",
        0,
    )

    if float(device_accounts) >= 3:
        evidence.append(
            "device has been associated with multiple accounts"
        )

    if network_score >= 7.0:
        evidence.append(
            "elevated network/entity risk"
        )

    if behavioral_signal >= 0.70:
        evidence.append(
            "behavior deviates materially from account history"
        )

    if model_probability >= 0.70:
        evidence.append(
            "model indicates elevated chargeback risk"
        )

    return evidence


def choose_primary_reason(
    decision: str,
    evidence: list[str],
    model_probability: float,
    network_score: float,
    behavioral_signal: float,
) -> str:
    """Choose one deterministic primary explanation."""

    if decision == "BLOCK":
        return (
            "high combined transaction and network risk"
        )

    if decision == "REVIEW":
        if network_score >= 7.0:
            return "elevated network/entity risk"

        if behavioral_signal >= 0.70:
            return (
                "material deviation from established behavior"
            )

        if model_probability >= 0.70:
            return (
                "elevated model-estimated transaction risk"
            )

        return "elevated combined risk"

    if evidence:
        return evidence[0]

    return "no significant risk signal detected"


def score_transaction(
    transaction: Mapping[str, Any],
    model_probability: float,
    network_score: float,
    behavioral_signal: float = 0.0,
) -> RiskCase:
    """
    End-to-end deterministic transaction risk decision.

    Pipeline:

        transaction
            -> evidence extraction
            -> risk fusion
            -> severity classification
            -> policy decision
            -> RiskCase
    """

    transaction_id = _transaction_value(
        transaction,
        "transaction_id",
    )

    if transaction_id is None:
        raise ValueError(
            "transaction must contain transaction_id"
        )

    model_probability = _clip_probability(
        model_probability
    )

    behavioral_signal = _clip_signal(
        behavioral_signal
    )

    network_score = max(
        float(network_score),
        0.0,
    )

    risk_score = fuse_risk(
        model_probability=model_probability,
        network_score=network_score,
        behavioral_signal=behavioral_signal,
    )

    level = risk_level(
        risk_score
    )

    decision = policy_decision(
        risk_score
    )

    evidence = build_evidence(
        transaction=transaction,
        model_probability=model_probability,
        network_score=network_score,
        behavioral_signal=behavioral_signal,
    )

    primary_reason = choose_primary_reason(
        decision=decision,
        evidence=evidence,
        model_probability=model_probability,
        network_score=network_score,
        behavioral_signal=behavioral_signal,
    )

    return RiskCase(
        transaction_id=str(transaction_id),
        risk_score=round(risk_score, 4),
        risk_level=level,
        decision=decision,
        primary_reason=primary_reason,
        evidence=evidence,
        model_probability=round(
            model_probability,
            6,
        ),
        network_score=round(
            network_score,
            4,
        ),
    )