from __future__ import annotations

from pathlib import Path

import pandas as pd

from razorguard.cases.queue import (
    cases_to_frame,
    rank_case_queue,
    summarize_case_queue,
    write_case_queue,
)
from razorguard.engine.pipeline import score_transaction


def make_case(
    transaction_id: str,
    probability: float,
    network: float,
    behavioral: float,
):
    return score_transaction(
        {
            "transaction_id": transaction_id,
            "amount": 1000,
            "location_mismatch": 1,
            "account_id_prior_count_60m": 4,
        },
        probability,
        network,
        behavioral,
    )


def test_cases_convert_to_frame():
    cases = [
        make_case("T001", 0.10, 1.0, 0.10),
        make_case("T002", 0.90, 10.0, 0.90),
    ]

    frame = cases_to_frame(cases)

    assert len(frame) == 2
    assert "evidence_text" in frame.columns
    assert frame["transaction_id"].tolist() == [
        "T001",
        "T002",
    ]


def test_queue_excludes_allowed_cases_by_default():
    cases = [
        make_case("LOW", 0.01, 0.0, 0.0),
        make_case("HIGH", 0.90, 10.0, 0.90),
    ]

    queue = rank_case_queue(cases)

    assert "LOW" not in queue["transaction_id"].tolist()
    assert "HIGH" in queue["transaction_id"].tolist()


def test_queue_can_include_allowed_cases():
    cases = [
        make_case("LOW", 0.01, 0.0, 0.0),
        make_case("HIGH", 0.90, 10.0, 0.90),
    ]

    queue = rank_case_queue(
        cases,
        include_allowed=True,
    )

    assert len(queue) == 2


def test_higher_risk_case_ranks_first():
    cases = [
        make_case("LOWER", 0.70, 5.0, 0.50),
        make_case("HIGHER", 0.95, 10.0, 0.90),
    ]

    queue = rank_case_queue(cases)

    assert queue.iloc[0]["transaction_id"] == "HIGHER"
    assert queue.iloc[0]["case_rank"] == 1


def test_block_ranks_before_review():
    cases = [
        make_case("REVIEW", 0.70, 5.0, 0.50),
        make_case("BLOCK", 1.0, 20.0, 1.0),
    ]

    queue = rank_case_queue(cases)

    assert queue.iloc[0]["transaction_id"] == "BLOCK"


def test_ranking_is_deterministic():
    cases = [
        make_case("B", 0.80, 8.0, 0.70),
        make_case("A", 0.80, 8.0, 0.70),
    ]

    first = rank_case_queue(cases)
    second = rank_case_queue(list(reversed(cases)))

    assert first["transaction_id"].tolist() == [
        "A",
        "B",
    ]

    assert first["transaction_id"].tolist() == (
        second["transaction_id"].tolist()
    )


def test_summary_contains_operational_counts():
    cases = [
        make_case("T001", 0.95, 20.0, 1.0),
        make_case("T002", 0.70, 5.0, 0.50),
    ]

    queue = rank_case_queue(cases)
    summary = summarize_case_queue(queue)

    assert summary["cases"] == 2
    assert summary["block_cases"] >= 1
    assert summary["mean_risk_score"] > 0


def test_empty_queue_summary():
    queue = rank_case_queue([])

    summary = summarize_case_queue(queue)

    assert summary["cases"] == 0
    assert summary["block_cases"] == 0
    assert summary["review_cases"] == 0


def test_queue_can_be_written_to_parquet(tmp_path: Path):
    cases = [
        make_case("T001", 0.90, 10.0, 0.90),
    ]

    queue = rank_case_queue(cases)

    output = tmp_path / "cases.parquet"

    write_case_queue(
        queue,
        str(output),
    )

    assert output.exists()

    restored = pd.read_parquet(output)

    assert restored["transaction_id"].tolist() == [
        "T001"
    ]

    assert "evidence_text" in restored.columns