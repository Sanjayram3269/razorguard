from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RiskCase:
    transaction_id: str
    risk_score: float
    risk_level: str
    decision: str
    primary_reason: str
    evidence: list[str]
    model_probability: float
    network_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)