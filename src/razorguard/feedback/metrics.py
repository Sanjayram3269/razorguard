from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from razorguard.feedback.outcomes import (
    CaseOutcome,
    is_actionable_feedback,
)


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return round(
        float(numerator) / float(denominator),
        6,
    )


def outcome_frame(
    outcomes: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Convert investigator outcomes into a normalized analytical frame.
    """

    rows = []

    for outcome in outcomes:
        row = dict(outcome)

        raw_outcome = row.get("outcome")

        if isinstance(raw_outcome, CaseOutcome):
            row["outcome"] = raw_outcome.value

        rows.append(row)

    columns = [
        "case_id",
        "transaction_id",
        "outcome",
        "confidence",
        "investigator",
        "notes",
        "created_at",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)

    for column in columns:
        if column not in frame.columns:
            frame[column] = None

    return frame[columns]


def compute_outcome_metrics(
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Compute investigator-feedback quality metrics.

    Only definitive investigator outcomes are used for evaluation.
    Escalated and insufficient-evidence outcomes are excluded from
    definitive precision/recall-style calculations.
    """

    frame = outcome_frame(outcomes)

    if frame.empty:
        return {
            "total_outcomes": 0,
            "actionable_outcomes": 0,
            "confirmed_fraud": 0,
            "legitimate": 0,
            "dismissed": 0,
            "escalated": 0,
            "insufficient_evidence": 0,
            "confirmation_rate": 0.0,
            "non_fraud_rate": 0.0,
        }

    counts = frame["outcome"].value_counts()

    confirmed = int(
        counts.get(
            CaseOutcome.CONFIRMED_FRAUD.value,
            0,
        )
    )

    legitimate = int(
        counts.get(
            CaseOutcome.LEGITIMATE.value,
            0,
        )
    )

    dismissed = int(
        counts.get(
            CaseOutcome.DISMISSED.value,
            0,
        )
    )

    escalated = int(
        counts.get(
            CaseOutcome.ESCALATED.value,
            0,
        )
    )

    insufficient = int(
        counts.get(
            CaseOutcome.INSUFFICIENT_EVIDENCE.value,
            0,
        )
    )

    actionable = confirmed + legitimate + dismissed

    non_fraud = legitimate + dismissed

    return {
        "total_outcomes": int(len(frame)),
        "actionable_outcomes": actionable,
        "confirmed_fraud": confirmed,
        "legitimate": legitimate,
        "dismissed": dismissed,
        "escalated": escalated,
        "insufficient_evidence": insufficient,
        "confirmation_rate": _safe_rate(
            confirmed,
            actionable,
        ),
        "non_fraud_rate": _safe_rate(
            non_fraud,
            actionable,
        ),
    }


def evaluate_decisions(
    cases: pd.DataFrame,
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Compare RazorGuard decisions with investigator outcomes.

    Investigator outcomes are joined by case_id. Only actionable
    outcomes participate in decision-quality evaluation.
    """

    if cases.empty:
        return {
            "evaluated_cases": 0,
            "confirmed_fraud": 0,
            "non_fraud": 0,
            "decision_precision": 0.0,
            "decision_recall": 0.0,
            "false_positive_rate": 0.0,
        }

    outcome_df = outcome_frame(outcomes)

    if outcome_df.empty:
        return {
            "evaluated_cases": 0,
            "confirmed_fraud": 0,
            "non_fraud": 0,
            "decision_precision": 0.0,
            "decision_recall": 0.0,
            "false_positive_rate": 0.0,
        }

    merged = cases.merge(
        outcome_df[
            [
                "case_id",
                "outcome",
            ]
        ],
        on="case_id",
        how="inner",
    )

    if merged.empty:
        return {
            "evaluated_cases": 0,
            "confirmed_fraud": 0,
            "non_fraud": 0,
            "decision_precision": 0.0,
            "decision_recall": 0.0,
            "false_positive_rate": 0.0,
        }

    merged = merged[
        merged["outcome"].isin(
            [
                CaseOutcome.CONFIRMED_FRAUD.value,
                CaseOutcome.LEGITIMATE.value,
                CaseOutcome.DISMISSED.value,
            ]
        )
    ].copy()

    if merged.empty:
        return {
            "evaluated_cases": 0,
            "confirmed_fraud": 0,
            "non_fraud": 0,
            "decision_precision": 0.0,
            "decision_recall": 0.0,
            "false_positive_rate": 0.0,
        }

    actual_positive = (
        merged["outcome"]
        == CaseOutcome.CONFIRMED_FRAUD.value
    )

    actual_negative = ~actual_positive

    predicted_positive = merged["decision"].isin(
        [
            "BLOCK",
            "REVIEW",
        ]
    )

    true_positive = int(
        (predicted_positive & actual_positive).sum()
    )

    false_positive = int(
        (predicted_positive & actual_negative).sum()
    )

    false_negative = int(
        (~predicted_positive & actual_positive).sum()
    )

    true_negative = int(
        (~predicted_positive & actual_negative).sum()
    )

    return {
        "evaluated_cases": int(len(merged)),
        "confirmed_fraud": int(actual_positive.sum()),
        "non_fraud": int(actual_negative.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "decision_precision": _safe_rate(
            true_positive,
            true_positive + false_positive,
        ),
        "decision_recall": _safe_rate(
            true_positive,
            true_positive + false_negative,
        ),
        "false_positive_rate": _safe_rate(
            false_positive,
            false_positive + true_negative,
        ),
    }


def decision_band_metrics(
    cases: pd.DataFrame,
    outcomes: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Produce outcome statistics grouped by RazorGuard decision band.
    """

    columns = [
        "decision",
        "evaluated_cases",
        "confirmed_fraud",
        "non_fraud",
        "confirmation_rate",
    ]

    if cases.empty:
        return pd.DataFrame(columns=columns)

    outcome_df = outcome_frame(outcomes)

    if outcome_df.empty:
        return pd.DataFrame(columns=columns)

    merged = cases.merge(
        outcome_df[
            [
                "case_id",
                "outcome",
            ]
        ],
        on="case_id",
        how="inner",
    )

    merged = merged[
        merged["outcome"].isin(
            [
                CaseOutcome.CONFIRMED_FRAUD.value,
                CaseOutcome.LEGITIMATE.value,
                CaseOutcome.DISMISSED.value,
            ]
        )
    ]

    if merged.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for decision, group in merged.groupby(
        "decision",
        sort=True,
    ):
        confirmed = int(
            (
                group["outcome"]
                == CaseOutcome.CONFIRMED_FRAUD.value
            ).sum()
        )

        non_fraud = int(len(group) - confirmed)

        rows.append(
            {
                "decision": decision,
                "evaluated_cases": int(len(group)),
                "confirmed_fraud": confirmed,
                "non_fraud": non_fraud,
                "confirmation_rate": _safe_rate(
                    confirmed,
                    len(group),
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )