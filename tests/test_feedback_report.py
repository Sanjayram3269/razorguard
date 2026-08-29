from __future__ import annotations

import json

import pandas as pd
import pytest

from razorguard.feedback.report import (
    build_feedback_report,
    build_store_feedback_report,
    load_feedback_report,
    write_feedback_report,
)
from razorguard.investigation.store import CaseStore


def sample_cases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CASE-T1",
                "CASE-T2",
                "CASE-T3",
            ],
            "transaction_id": [
                "T1",
                "T2",
                "T3",
            ],
            "decision": [
                "BLOCK",
                "REVIEW",
                "REVIEW",
            ],
            "risk_score": [
                95.0,
                72.0,
                61.0,
            ],
        }
    )


def sample_outcomes() -> list[dict[str, object]]:
    return [
        {
            "case_id": "CASE-T1",
            "transaction_id": "T1",
            "outcome": "confirmed_fraud",
            "confidence": "high",
            "investigator": "analyst-01",
        },
        {
            "case_id": "CASE-T2",
            "transaction_id": "T2",
            "outcome": "legitimate",
            "confidence": "high",
            "investigator": "analyst-02",
        },
        {
            "case_id": "CASE-T3",
            "transaction_id": "T3",
            "outcome": "confirmed_fraud",
            "confidence": "medium",
            "investigator": "analyst-01",
        },
    ]


def test_build_feedback_report_contains_all_sections():
    report = build_feedback_report(
        sample_cases(),
        sample_outcomes(),
    )

    assert report["report_version"] == "e2.6"
    assert "outcomes" in report
    assert "decision_quality" in report
    assert "decision_bands" in report


def test_report_contains_outcome_metrics():
    report = build_feedback_report(
        sample_cases(),
        sample_outcomes(),
    )

    assert report["outcomes"]["total_outcomes"] == 3
    assert report["outcomes"]["actionable_outcomes"] == 3
    assert report["outcomes"]["confirmed_fraud"] == 2
    assert report["outcomes"]["legitimate"] == 1


def test_report_contains_decision_quality():
    report = build_feedback_report(
        sample_cases(),
        sample_outcomes(),
    )

    metrics = report["decision_quality"]

    assert metrics["evaluated_cases"] == 3
    assert metrics["confirmed_fraud"] == 2
    assert metrics["non_fraud"] == 1
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1


def test_report_contains_decision_bands():
    report = build_feedback_report(
        sample_cases(),
        sample_outcomes(),
    )

    bands = report["decision_bands"]

    assert len(bands) == 2
    assert {
        row["decision"]
        for row in bands
    } == {
        "BLOCK",
        "REVIEW",
    }


def test_report_handles_empty_inputs():
    report = build_feedback_report(
        pd.DataFrame(),
        [],
    )

    assert report["outcomes"]["total_outcomes"] == 0
    assert report["decision_quality"]["evaluated_cases"] == 0
    assert report["decision_bands"] == []


def test_write_and_load_feedback_report(tmp_path):
    report = build_feedback_report(
        sample_cases(),
        sample_outcomes(),
    )

    path = tmp_path / "feedback_report.json"

    written = write_feedback_report(
        report,
        path,
    )

    assert written == path
    assert path.exists()

    loaded = load_feedback_report(path)

    assert loaded == report


def test_load_missing_report_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feedback_report(
            tmp_path / "missing.json"
        )


def test_store_feedback_report(tmp_path):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    store.create(
        {
            "transaction_id": "T1",
            "risk_score": 95.0,
            "risk_level": "CRITICAL",
            "decision": "BLOCK",
            "primary_reason": "high risk",
            "evidence": ["signal"],
            "model_probability": 0.95,
            "network_score": 10.0,
        }
    )

    outcomes = [
        {
            "case_id": "CASE-T1",
            "transaction_id": "T1",
            "outcome": "confirmed_fraud",
            "confidence": "high",
            "investigator": "analyst-01",
        }
    ]

    report = build_store_feedback_report(
        store,
        outcomes,
    )

    assert report["outcomes"]["total_outcomes"] == 1
    assert report["decision_quality"]["evaluated_cases"] == 1
    assert report["decision_quality"]["true_positive"] == 1