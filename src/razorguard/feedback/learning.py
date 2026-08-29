from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


DEFINITIVE_OUTCOMES = {
    "confirmed_fraud",
    "legitimate",
    "dismissed",
}


def _value(
    row: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    return row.get(key, default)


def _normalize_outcome(
    outcome: Any,
) -> str:
    return str(outcome or "").strip().lower()


def _is_fraud_outcome(
    outcome: str,
) -> bool:
    return outcome == "confirmed_fraud"


def _is_definitive(
    outcome: str,
) -> bool:
    return outcome in DEFINITIVE_OUTCOMES


def build_learning_dataset(
    cases: pd.DataFrame,
    outcomes: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Build an auditable investigator-feedback learning dataset.

    Each row joins a scored case with its investigator outcome.

    No automatic retraining occurs here. The output is intended to
    become a controlled learning/calibration input for later model
    iterations.
    """

    outcome_rows = list(outcomes)

    columns = [
        "case_id",
        "transaction_id",
        "risk_score",
        "risk_level",
        "decision",
        "model_probability",
        "network_score",
        "outcome",
        "confidence",
        "is_definitive",
        "actual_fraud",
        "decision_correct",
        "error_type",
        "learning_signal",
    ]

    if cases.empty or not outcome_rows:
        return pd.DataFrame(columns=columns)

    case_frame = cases.copy()

    if "case_id" not in case_frame.columns:
        if "transaction_id" not in case_frame.columns:
            return pd.DataFrame(columns=columns)

        case_frame["case_id"] = (
            "CASE-"
            + case_frame["transaction_id"].astype(str)
        )

    outcome_frame = pd.DataFrame(
        outcome_rows
    )

    if outcome_frame.empty:
        return pd.DataFrame(columns=columns)

    required_outcome_columns = {
        "transaction_id",
        "outcome",
    }

    if not required_outcome_columns.issubset(
        outcome_frame.columns
    ):
        return pd.DataFrame(columns=columns)

    if "case_id" not in outcome_frame.columns:
        outcome_frame["case_id"] = (
            "CASE-"
            + outcome_frame["transaction_id"].astype(str)
        )

    outcome_frame["outcome"] = (
        outcome_frame["outcome"]
        .map(_normalize_outcome)
    )

    merged = case_frame.merge(
        outcome_frame[
            [
                column
                for column in [
                    "case_id",
                    "transaction_id",
                    "outcome",
                    "confidence",
                ]
                if column in outcome_frame.columns
            ]
        ],
        on=[
            "case_id",
            "transaction_id",
        ],
        how="inner",
        suffixes=("", "_outcome"),
    )

    if merged.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        outcome = _normalize_outcome(
            _value(row, "outcome")
        )

        decision = str(
            _value(row, "decision", "")
        ).upper()

        definitive = _is_definitive(
            outcome
        )

        actual_fraud = (
            _is_fraud_outcome(outcome)
            if definitive
            else None
        )

        decision_correct = None
        error_type = "UNRESOLVED"
        learning_signal = "UNRESOLVED"

        if definitive:
            if actual_fraud:
                if decision in {
                    "BLOCK",
                    "REVIEW",
                }:
                    decision_correct = True
                    error_type = "TRUE_POSITIVE"
                    learning_signal = "CORRECT_FRAUD_DETECTION"
                else:
                    decision_correct = False
                    error_type = "FALSE_NEGATIVE"
                    learning_signal = "MISSED_FRAUD"
            else:
                if decision == "ALLOW":
                    decision_correct = True
                    error_type = "TRUE_NEGATIVE"
                    learning_signal = "CORRECT_LEGITIMATE_DECISION"
                else:
                    decision_correct = False
                    error_type = "FALSE_POSITIVE"
                    learning_signal = "EXCESSIVE_INTERVENTION"

        rows.append(
            {
                "case_id": str(
                    _value(row, "case_id")
                ),
                "transaction_id": str(
                    _value(row, "transaction_id")
                ),
                "risk_score": float(
                    _value(row, "risk_score", 0.0)
                ),
                "risk_level": str(
                    _value(row, "risk_level", "")
                ),
                "decision": decision,
                "model_probability": float(
                    _value(
                        row,
                        "model_probability",
                        0.0,
                    )
                ),
                "network_score": float(
                    _value(
                        row,
                        "network_score",
                        0.0,
                    )
                ),
                "outcome": outcome,
                "confidence": _value(
                    row,
                    "confidence",
                ),
                "is_definitive": definitive,
                "actual_fraud": actual_fraud,
                "decision_correct": decision_correct,
                "error_type": error_type,
                "learning_signal": learning_signal,
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def summarize_learning_dataset(
    learning_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Produce compact learning-quality statistics.
    """

    if learning_data.empty:
        return {
            "rows": 0,
            "definitive_rows": 0,
            "unresolved_rows": 0,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "decision_accuracy": 0.0,
        }

    definitive = learning_data[
        learning_data["is_definitive"]
    ]

    correct = definitive[
        definitive["decision_correct"] == True
    ]

    return {
        "rows": int(len(learning_data)),
        "definitive_rows": int(
            len(definitive)
        ),
        "unresolved_rows": int(
            len(learning_data)
            - len(definitive)
        ),
        "true_positive": int(
            (
                learning_data["error_type"]
                == "TRUE_POSITIVE"
            ).sum()
        ),
        "true_negative": int(
            (
                learning_data["error_type"]
                == "TRUE_NEGATIVE"
            ).sum()
        ),
        "false_positive": int(
            (
                learning_data["error_type"]
                == "FALSE_POSITIVE"
            ).sum()
        ),
        "false_negative": int(
            (
                learning_data["error_type"]
                == "FALSE_NEGATIVE"
            ).sum()
        ),
        "decision_accuracy": round(
            float(
                len(correct)
                / len(definitive)
            )
            if len(definitive)
            else 0.0,
            6,
        ),
    }


def learning_error_breakdown(
    learning_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate learning errors by decision and risk level.

    This helps identify where policy/model calibration is weakest.
    """

    columns = [
        "decision",
        "risk_level",
        "cases",
        "correct",
        "false_positive",
        "false_negative",
        "accuracy",
    ]

    if learning_data.empty:
        return pd.DataFrame(
            columns=columns
        )

    definitive = learning_data[
        learning_data["is_definitive"]
    ].copy()

    if definitive.empty:
        return pd.DataFrame(
            columns=columns
        )

    grouped = (
        definitive
        .groupby(
            [
                "decision",
                "risk_level",
            ],
            dropna=False,
        )
        .agg(
            cases=("case_id", "count"),
            correct=(
                "decision_correct",
                "sum",
            ),
            false_positive=(
                "error_type",
                lambda values: (
                    values == "FALSE_POSITIVE"
                ).sum(),
            ),
            false_negative=(
                "error_type",
                lambda values: (
                    values == "FALSE_NEGATIVE"
                ).sum(),
            ),
        )
        .reset_index()
    )

    grouped["accuracy"] = (
        grouped["correct"]
        / grouped["cases"]
    ).round(6)

    return grouped[
        columns
    ].sort_values(
        [
            "decision",
            "risk_level",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def generate_learning_signals(
    learning_data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert investigator outcomes into explicit calibration signals.

    These are recommendations only; they do not mutate the model.
    """

    if learning_data.empty:
        return []

    definitive = learning_data[
        learning_data["is_definitive"]
    ]

    signals: list[dict[str, Any]] = []

    false_positive_count = int(
        (
            definitive["error_type"]
            == "FALSE_POSITIVE"
        ).sum()
    )

    false_negative_count = int(
        (
            definitive["error_type"]
            == "FALSE_NEGATIVE"
        ).sum()
    )

    if false_positive_count:
        signals.append(
            {
                "signal": "REDUCE_FALSE_POSITIVES",
                "count": false_positive_count,
                "recommendation": (
                    "Review intervention thresholds and "
                    "risk features associated with legitimate cases."
                ),
            }
        )

    if false_negative_count:
        signals.append(
            {
                "signal": "REDUCE_FALSE_NEGATIVES",
                "count": false_negative_count,
                "recommendation": (
                    "Review missed-fraud cases for weak behavioral "
                    "or network signals and calibration gaps."
                ),
            }
        )

    return signals