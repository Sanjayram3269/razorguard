from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from razorguard.feedback.metrics import (
    compute_outcome_metrics,
    decision_band_metrics,
    evaluate_decisions,
)
from razorguard.feedback.outcomes import CaseOutcome
from razorguard.investigation.store import CaseStore


def _load_outcomes(
    outcomes_path: str | Path | None,
) -> list[dict[str, Any]]:
    """Load persisted investigator outcomes."""

    if outcomes_path is None:
        return []

    path = Path(outcomes_path)

    if not path.exists():
        return []

    frame = pd.read_parquet(path)

    if frame.empty:
        return []

    return frame.to_dict(orient="records")


def build_feedback_report(
    cases: pd.DataFrame,
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete investigator-feedback intelligence report.

    The report combines:
        - outcome distribution
        - decision-quality metrics
        - decision-band performance

    Non-definitive outcomes are retained in operational counts but
    excluded from definitive evaluation metrics.
    """

    outcome_list = list(outcomes)

    outcome_metrics = compute_outcome_metrics(
        outcome_list
    )

    decision_metrics = evaluate_decisions(
        cases,
        outcome_list,
    )

    band_frame = decision_band_metrics(
        cases,
        outcome_list,
    )

    decision_bands = (
        band_frame.to_dict(orient="records")
        if not band_frame.empty
        else []
    )

    return {
        "report_version": "e2.6",
        "outcomes": outcome_metrics,
        "decision_quality": decision_metrics,
        "decision_bands": decision_bands,
    }


def build_store_feedback_report(
    store: CaseStore,
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Generate feedback intelligence directly from a CaseStore.
    """

    cases = store.list()

    return build_feedback_report(
        cases=cases,
        outcomes=outcomes,
    )


def write_feedback_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Persist a feedback report as deterministic JSON."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            dict(report),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return output_path


def load_feedback_report(
    path: str | Path,
) -> dict[str, Any]:
    """Load a previously generated feedback report."""

    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"feedback report not found: {report_path}"
        )

    return json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )