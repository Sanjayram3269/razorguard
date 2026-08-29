from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import pandas as pd

from razorguard.risk.case import RiskCase


def _value(
    row: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    return row.get(key, default)


def _number(
    row: Mapping[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    try:
        value = _value(row, key, default)
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _flag(
    row: Mapping[str, Any],
    key: str,
) -> bool:
    return bool(_number(row, key, 0.0))


def build_account_context(
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract point-in-time account context available for investigation.

    No future labels or post-event information are used.
    """

    return {
        "account_id": _value(transaction, "account_id"),
        "account_age_days": round(
            _number(transaction, "account_age_days"),
            2,
        ),
        "prior_transaction_count": int(
            _number(transaction, "prior_tx_count")
        ),
        "prior_count_60m": int(
            _number(
                transaction,
                "account_id_prior_count_60m",
            )
        ),
        "prior_count_1440m": int(
            _number(
                transaction,
                "account_id_prior_count_1440m",
            )
        ),
        "prior_unique_merchants": int(
            _number(
                transaction,
                "prior_unique_merchants",
            )
        ),
        "dormant_return": _flag(
            transaction,
            "is_dormant_return",
        ),
    }


def build_network_context(
    transaction: Mapping[str, Any],
    network_score: float,
) -> dict[str, Any]:
    """
    Extract relationship evidence available before the transaction.
    """

    return {
        "device_id": _value(
            transaction,
            "device_id",
        ),
        "merchant_id": _value(
            transaction,
            "merchant_id",
        ),
        "prior_accounts_per_device": int(
            _number(
                transaction,
                "prior_accounts_per_device",
            )
        ),
        "prior_accounts_per_merchant": int(
            _number(
                transaction,
                "prior_accounts_per_merchant",
            )
        ),
        "prior_merchants_per_device": int(
            _number(
                transaction,
                "prior_merchants_per_device",
            )
        ),
        "account_device_novelty": int(
            _number(
                transaction,
                "account_device_novelty",
            )
        ),
        "account_merchant_novelty": int(
            _number(
                transaction,
                "account_merchant_novelty",
            )
        ),
        "network_score": round(
            max(float(network_score), 0.0),
            4,
        ),
    }


def build_behavior_context(
    transaction: Mapping[str, Any],
    behavioral_signal: float,
) -> dict[str, Any]:
    """
    Summarize observable behavioral deviation signals.
    """

    return {
        "behavioral_signal": round(
            max(
                0.0,
                min(float(behavioral_signal), 1.0),
            ),
            6,
        ),
        "amount_zscore": round(
            _number(transaction, "amount_zscore"),
            4,
        ),
        "velocity_ratio": round(
            _number(
                transaction,
                "account_velocity_ratio",
            ),
            4,
        ),
        "location_mismatch": _flag(
            transaction,
            "location_mismatch",
        ),
        "dormant_return": _flag(
            transaction,
            "is_dormant_return",
        ),
    }


def build_investigation_narrative(
    case: RiskCase,
    account_context: Mapping[str, Any],
    network_context: Mapping[str, Any],
    behavior_context: Mapping[str, Any],
) -> str:
    """
    Produce a concise deterministic investigator narrative.

    The narrative describes evidence rather than claiming certainty
    about fraud.
    """

    parts: list[str] = []

    parts.append(
        f"Transaction {case.transaction_id} is classified "
        f"as {case.risk_level} risk with a {case.decision} decision."
    )

    if case.primary_reason:
        parts.append(
            f"Primary signal: {case.primary_reason}."
        )

    if account_context.get("prior_count_60m", 0) >= 3:
        parts.append(
            "Recent account velocity is elevated."
        )

    if account_context.get("dormant_return"):
        parts.append(
            "The transaction follows an extended inactive period."
        )

    if behavior_context.get("location_mismatch"):
        parts.append(
            "The observed IP and shipping countries differ."
        )

    if behavior_context.get("amount_zscore", 0.0) >= 2.0:
        parts.append(
            "Transaction amount is materially above the "
            "account's recent baseline."
        )

    if network_context.get(
        "prior_accounts_per_device",
        0,
    ) >= 3:
        parts.append(
            "The device has prior associations with multiple accounts."
        )

    if network_context.get(
        "prior_accounts_per_merchant",
        0,
    ) >= 20:
        parts.append(
            "The merchant has substantial prior account exposure."
        )

    if behavior_context.get(
        "behavioral_signal",
        0.0,
    ) >= 0.70:
        parts.append(
            "Observed behavior materially deviates from the "
            "account's historical pattern."
        )

    parts.append(
        f"Model probability is "
        f"{case.model_probability:.1%}; network risk is "
        f"{case.network_score:.2f}."
    )

    return " ".join(parts)


def enrich_case(
    case: RiskCase,
    transaction: Mapping[str, Any],
    behavioral_signal: float = 0.0,
) -> dict[str, Any]:
    """
    Build a complete investigator-facing case artifact.

    All enrichment is derived from the transaction state available
    at scoring time.
    """

    account_context = build_account_context(
        transaction
    )

    network_context = build_network_context(
        transaction,
        case.network_score,
    )

    behavior_context = build_behavior_context(
        transaction,
        behavioral_signal,
    )

    narrative = build_investigation_narrative(
        case=case,
        account_context=account_context,
        network_context=network_context,
        behavior_context=behavior_context,
    )

    return {
        **asdict(case),
        "account_context": account_context,
        "network_context": network_context,
        "behavior_context": behavior_context,
        "investigation_narrative": narrative,
    }