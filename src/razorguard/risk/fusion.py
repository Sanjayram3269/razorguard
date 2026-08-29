from __future__ import annotations

import numpy as np


def normalize_network_score(
    network_score: float,
    ceiling: float = 10.0,
) -> float:
    """Map network risk into [0, 1] with saturation."""
    value = max(float(network_score), 0.0)
    return float(
        np.clip(
            value / ceiling,
            0.0,
            1.0,
        )
    )


def fuse_risk(
    model_probability: float,
    network_score: float,
    behavioral_signal: float = 0.0,
) -> float:
    """
    Combine independent evidence channels into a bounded 0-100 risk score.

    The fusion weights are explicit and deterministic.
    """

    model_probability = float(
        np.clip(model_probability, 0.0, 1.0)
    )

    network_probability = normalize_network_score(
        network_score
    )

    behavioral_signal = float(
        np.clip(behavioral_signal, 0.0, 1.0)
    )

    score = (
        0.55 * model_probability
        + 0.30 * network_probability
        + 0.15 * behavioral_signal
    )

    return float(
        np.clip(score * 100.0, 0.0, 100.0)
    )


def risk_level(score: float) -> str:
    """Map continuous risk into an explainable severity band."""

    score = float(score)

    if score >= 85:
        return "CRITICAL"

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"