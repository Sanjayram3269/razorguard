from __future__ import annotations

import json

import pandas as pd

from razorguard.feedback.artifacts import (
    write_feedback_artifacts,
)


def test_write_feedback_artifacts(tmp_path):
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-T1",
                "CASE-T2",
            ],
            "transaction_id": [
                "T1",
                "T2",
            ],
            "risk_score": [
                95.0,
                20.0,
            ],
            "risk_level": [
                "CRITICAL",
                "LOW",
            ],
            "decision": [
                "BLOCK",
                "ALLOW",
            ],
            "model_probability": [
                0.95,
                0.02,
            ],
            "network_score": [
                10.0,
                0.0,
            ],
        }
    )

    outcomes = [
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
    ]

    paths = write_feedback_artifacts(
        cases,
        outcomes,
        tmp_path,
    )

    assert set(paths) == {
        "feedback_report",
        "learning_dataset",
        "calibration_signals",
    }

    report_path = tmp_path / "feedback_report.json"
    learning_path = (
        tmp_path / "learning_dataset.parquet"
    )
    signals_path = (
        tmp_path / "calibration_signals.json"
    )

    assert report_path.exists()
    assert learning_path.exists()
    assert signals_path.exists()

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["report_version"] == "e2.6"
    assert report["learning"]["rows"] == 2
    assert report["learning"]["definitive_rows"] == 2
    assert report["learning"]["true_positive"] == 1
    assert report["learning"]["true_negative"] == 1

    learning = pd.read_parquet(
        learning_path
    )

    assert len(learning) == 2
    assert set(
        learning["error_type"]
    ) == {
        "TRUE_POSITIVE",
        "TRUE_NEGATIVE",
    }

    signals = json.loads(
        signals_path.read_text(
            encoding="utf-8"
        )
    )

    assert signals["version"] == "e2.6"
    assert signals["signals"] == []


def test_artifact_generation_creates_directory(
    tmp_path,
):
    cases = pd.DataFrame(
        {
            "case_id": ["CASE-T1"],
            "transaction_id": ["T1"],
            "risk_score": [90.0],
            "risk_level": ["CRITICAL"],
            "decision": ["BLOCK"],
            "model_probability": [0.9],
            "network_score": [10.0],
        }
    )

    outcomes = [
        {
            "case_id": "CASE-T1",
            "transaction_id": "T1",
            "outcome": "legitimate",
        }
    ]

    output = (
        tmp_path
        / "nested"
        / "feedback"
    )

    paths = write_feedback_artifacts(
        cases,
        outcomes,
        output,
    )

    assert output.exists()

    assert all(
        path
        for path in paths.values()
    )


def test_artifact_generation_handles_empty_data(
    tmp_path,
):
    paths = write_feedback_artifacts(
        pd.DataFrame(),
        [],
        tmp_path,
    )

    assert (
        tmp_path / "feedback_report.json"
    ).exists()

    assert (
        tmp_path / "learning_dataset.parquet"
    ).exists()

    assert (
        tmp_path / "calibration_signals.json"
    ).exists()