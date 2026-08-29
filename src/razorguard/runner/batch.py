from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from razorguard.cases.enrichment import enrich_case
from razorguard.cases.queue import (
    rank_case_queue,
    summarize_case_queue,
    write_case_queue,
)
from razorguard.config import ARTIFACTS, TRANSACTIONS_PATH
from razorguard.engine.pipeline import score_transaction
from razorguard.graph.risk import add_network_risk_features
from razorguard.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_frame,
)


MODEL_PATH = ARTIFACTS / "risk_model.joblib"
METADATA_PATH = ARTIFACTS / "model_metadata.json"

SCORED_PATH = ARTIFACTS / "scored_transactions.parquet"
CASES_PATH = ARTIFACTS / "investigator_cases.parquet"
SUMMARY_PATH = ARTIFACTS / "run_summary.json"


def load_model() -> tuple[Any, dict[str, Any]]:
    """Load the trained risk model and its metadata."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Risk model not found: {MODEL_PATH}. "
            "Run `python -m razorguard.ml.train` first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}. "
            "Run `python -m razorguard.ml.train` first."
        )

    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    return model, metadata


def behavioral_signal(row: pd.Series) -> float:
    """
    Convert observable behavioral deviations into [0, 1].

    This is deliberately deterministic and independent of the
    supervised model probability.
    """

    amount_zscore = float(
        row.get("amount_zscore", 0.0)
    )

    dormant_return = float(
        row.get("is_dormant_return", 0)
    )

    velocity_60m = float(
        row.get(
            "account_id_prior_count_60m",
            0,
        )
    )

    location_mismatch = float(
        row.get(
            "location_mismatch",
            0,
        )
    )

    velocity_ratio = float(
        row.get(
            "account_velocity_ratio",
            0.0,
        )
    )

    signals = [
        min(
            amount_zscore / 4.0,
            1.0,
        )
        if amount_zscore > 0
        else 0.0,

        1.0
        if dormant_return
        else 0.0,

        min(
            velocity_60m / 8.0,
            1.0,
        ),

        1.0
        if location_mismatch
        else 0.0,

        min(
            velocity_ratio,
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


def score_batch(
    transactions: pd.DataFrame,
    model: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score a complete transaction batch.

    Pipeline:

        transactions
            -> point-in-time model features
            -> network features
            -> model probability
            -> behavioral signal
            -> deterministic risk fusion
            -> investigator enrichment
            -> ranked case queue

    Returns:
        scored_transactions
        investigator_queue
    """

    model_frame = build_model_frame(
        transactions
    )

    network_frame = add_network_risk_features(
        transactions
    )

    model_input = model_frame[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    model_probability = (
    pd.DataFrame(
        model.predict_proba(model_input)
    )
    .iloc[:, 1]
    .to_numpy(dtype=float)
)

    network_scores = (
        network_frame[
            "network_risk_score"
        ]
        .to_numpy(dtype=float)
    )

    cases = []
    enriched_cases = []
    scored_rows = []

    for index, row in model_frame.iterrows():
        transaction = row.to_dict()

        network_score = float(
            network_scores[index]
        )

        probability = float(
            model_probability[index]
        )

        behavior = behavioral_signal(
            row
        )

        case = score_transaction(
            transaction=transaction,
            model_probability=probability,
            network_score=network_score,
            behavioral_signal=behavior,
        )

        enriched_case = enrich_case(
            case=case,
            transaction=transaction,
            behavioral_signal=behavior,
        )

        # Keep the canonical RiskCase separately.
        # rank_case_queue() expects RiskCase objects.
        cases.append(case)

        # Keep the complete investigator-facing artifact.
        enriched_cases.append(
            enriched_case
        )

        scored_rows.append(
            {
                "transaction_id": case.transaction_id,
                "model_probability": (
                    case.model_probability
                ),
                "network_score": (
                    case.network_score
                ),
                "behavioral_signal": round(
                    behavior,
                    6,
                ),
                "risk_score": case.risk_score,
                "risk_level": case.risk_level,
                "decision": case.decision,
                "primary_reason": (
                    case.primary_reason
                ),
                "evidence_text": " | ".join(
                    case.evidence
                ),
                "investigation_narrative": (
                    enriched_case[
                        "investigation_narrative"
                    ]
                ),
            }
        )

    scored = pd.DataFrame(
        scored_rows
    )

    queue = rank_case_queue(
        cases
    )

    return scored, queue


def run_batch(
    transactions_path: Path = TRANSACTIONS_PATH,
    artifacts_path: Path = ARTIFACTS,
) -> dict[str, Any]:
    """
    Execute the complete offline RazorGuard risk pipeline.
    """

    artifacts_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, metadata = load_model()

    transactions = pd.read_parquet(
        transactions_path
    )

    scored, queue = score_batch(
        transactions,
        model,
    )

    scored_path = (
        artifacts_path
        / "scored_transactions.parquet"
    )

    cases_path = (
        artifacts_path
        / "investigator_cases.parquet"
    )

    summary_path = (
        artifacts_path
        / "run_summary.json"
    )

    scored.to_parquet(
        scored_path,
        index=False,
    )

    write_case_queue(
        queue,
        str(cases_path),
    )

    queue_summary = summarize_case_queue(
        queue
    )

    decision_counts = (
        scored["decision"]
        .value_counts()
        .to_dict()
    )

    level_counts = (
        scored["risk_level"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "transactions": int(
            len(transactions)
        ),
        "scored_transactions": int(
            len(scored)
        ),
        "investigator_cases": int(
            len(queue)
        ),
        "model": metadata.get(
            "model"
        ),
        "threshold": metadata.get(
            "threshold"
        ),
        "decision_counts": {
            str(key): int(value)
            for key, value in decision_counts.items()
        },
        "risk_level_counts": {
            str(key): int(value)
            for key, value in level_counts.items()
        },
        "case_queue": queue_summary,
        "artifacts": {
            "scored_transactions": str(
                scored_path
            ),
            "investigator_cases": str(
                cases_path
            ),
            "run_summary": str(
                summary_path
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    summary = run_batch()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()