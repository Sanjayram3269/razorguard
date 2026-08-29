from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from razorguard.feedback.learning import (
    build_learning_dataset,
    generate_learning_signals,
    summarize_learning_dataset,
)
from razorguard.feedback.report import build_feedback_report


def write_feedback_artifacts(
    cases: pd.DataFrame,
    outcomes: Iterable[Mapping[str, Any]],
    artifacts_path: str | Path,
) -> dict[str, str]:
    """
    Generate the complete E2.6 feedback artifact bundle.

    Produces:
        feedback_report.json
        learning_dataset.parquet
        calibration_signals.json
    """

    output_dir = Path(artifacts_path)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outcome_list = list(outcomes)

    report = build_feedback_report(
        cases=cases,
        outcomes=outcome_list,
    )

    learning_data = build_learning_dataset(
        cases=cases,
        outcomes=outcome_list,
    )

    learning_summary = summarize_learning_dataset(
        learning_data
    )

    signals = generate_learning_signals(
        learning_data
    )

    report["learning"] = learning_summary

    report_path = (
        output_dir / "feedback_report.json"
    )

    learning_path = (
        output_dir / "learning_dataset.parquet"
    )

    signals_path = (
        output_dir / "calibration_signals.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    learning_data.to_parquet(
        learning_path,
        index=False,
    )

    signals_path.write_text(
        json.dumps(
            {
                "version": "e2.6",
                "signals": signals,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "feedback_report": str(
            report_path
        ),
        "learning_dataset": str(
            learning_path
        ),
        "calibration_signals": str(
            signals_path
        ),
    }