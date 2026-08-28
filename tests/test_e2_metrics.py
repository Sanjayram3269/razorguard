from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = Path("artifacts/test_metrics.json")


def test_metric_artifact_is_well_formed():
    if not ARTIFACT.exists():
        return

    metrics = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    for key in ["precision", "recall", "f1", "false_positive_rate"]:
        assert 0.0 <= metrics[key] <= 1.0

    assert metrics["test_rows"] > 0
    assert metrics["true_negative"] >= 0
    assert metrics["false_positive"] >= 0
    assert metrics["false_negative"] >= 0
    assert metrics["true_positive"] >= 0
