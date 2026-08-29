from __future__ import annotations
from unittest import result

import pandas as pd

from razorguard.feedback.learning import (
    build_learning_dataset,
    generate_learning_signals,
    learning_error_breakdown,
    summarize_learning_dataset,
)


def cases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CASE-T1",
                "CASE-T2",
                "CASE-T3",
                "CASE-T4",
            ],
            "transaction_id": [
                "T1",
                "T2",
                "T3",
                "T4",
            ],
            "risk_score": [
                95.0,
                20.0,
                75.0,
                30.0,
            ],
            "risk_level": [
                "CRITICAL",
                "LOW",
                "HIGH",
                "LOW",
            ],
            "decision": [
                "BLOCK",
                "ALLOW",
                "REVIEW",
                "ALLOW",
            ],
            "model_probability": [
                0.95,
                0.02,
                0.80,
                0.10,
            ],
            "network_score": [
                10.0,
                0.0,
                8.0,
                0.0,
            ],
        }
    )


def outcomes():
    return [
        {
            "case_id": "CASE-T1",
            "transaction_id": "T1",
            "outcome": "confirmed_fraud",
            "confidence": "high",
        },
        {
            "case_id": "CASE-T2",
            "transaction_id": "T2",
            "outcome": "legitimate",
            "confidence": "high",
        },
        {
            "case_id": "CASE-T3",
            "transaction_id": "T3",
            "outcome": "legitimate",
            "confidence": "high",
        },
        {
            "case_id": "CASE-T4",
            "transaction_id": "T4",
            "outcome": "confirmed_fraud",
            "confidence": "high",
        },
    ]


def test_build_learning_dataset_classifies_outcomes():
    result = build_learning_dataset(
        cases(),
        outcomes(),
    )

    assert len(result) == 4

    assert (
        result["error_type"].tolist()
        == [
            "TRUE_POSITIVE",
            "TRUE_NEGATIVE",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ]
    )


def test_learning_dataset_contains_only_definitive_cases():
    result = build_learning_dataset(
        cases(),
        [
            {
                "case_id": "CASE-T1",
                "transaction_id": "T1",
                "outcome": "confirmed_fraud",
            },
            {
                "case_id": "CASE-T2",
                "transaction_id": "T2",
                "outcome": "escalated",
            },
        ],
    )

    assert len(result) == 2
    assert bool(result.iloc[0]["is_definitive"]) is True
    assert bool(result.iloc[1]["is_definitive"]) is False


def test_learning_summary():
    result = build_learning_dataset(
        cases(),
        outcomes(),
    )

    summary = summarize_learning_dataset(
        result
    )

    assert summary["rows"] == 4
    assert summary["definitive_rows"] == 4
    assert summary["true_positive"] == 1
    assert summary["true_negative"] == 1
    assert summary["false_positive"] == 1
    assert summary["false_negative"] == 1
    assert summary["decision_accuracy"] == 0.5


def test_learning_error_breakdown():
    result = build_learning_dataset(
        cases(),
        outcomes(),
    )

    breakdown = learning_error_breakdown(
        result
    )

    assert len(breakdown) == 3
    assert "false_positive" in breakdown.columns
    assert "false_negative" in breakdown.columns
    assert "accuracy" in breakdown.columns


def test_generate_learning_signals():
    result = build_learning_dataset(
        cases(),
        outcomes(),
    )

    signals = generate_learning_signals(
        result
    )

    names = {
        signal["signal"]
        for signal in signals
    }

    assert "REDUCE_FALSE_POSITIVES" in names
    assert "REDUCE_FALSE_NEGATIVES" in names


def test_empty_learning_dataset():
    result = build_learning_dataset(
        pd.DataFrame(),
        [],
    )

    summary = summarize_learning_dataset(
        result
    )

    assert summary["rows"] == 0
    assert summary["definitive_rows"] == 0

    assert learning_error_breakdown(
        result
    ).empty

    assert generate_learning_signals(
        result
    ) == []