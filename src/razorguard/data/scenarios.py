from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    normal: float = 0.56
    new_account: float = 0.10
    high_value_legitimate: float = 0.08
    compromised: float = 0.12
    burst: float = 0.07
    coordinated: float = 0.05
    shared_infrastructure: float = 0.02

    @property
    def probabilities(self) -> list[float]:
        values = [
            self.normal,
            self.new_account,
            self.high_value_legitimate,
            self.compromised,
            self.burst,
            self.coordinated,
            self.shared_infrastructure,
        ]
        total = sum(values)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Scenario probabilities must sum to 1.0, got {total}")
        return values


SCENARIOS = np.array(
    [
        "normal",
        "new_account",
        "high_value_legitimate",
        "compromised",
        "burst",
        "coordinated",
        "shared_infrastructure",
    ]
)


def assign_account_scenarios(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    config: ScenarioConfig | None = None,
) -> pd.DataFrame:
    """Assign one behavioral archetype to every account.

    The scenario is simulation metadata. It is deliberately not consumed by
    the runtime model; it exists to make the synthetic world auditable.
    """
    config = config or ScenarioConfig()
    out = accounts.copy()
    out["scenario"] = rng.choice(
        SCENARIOS,
        size=len(out),
        p=config.probabilities,
    )

    # Make account age consistent with the scenario rather than using a
    # random segment as a proxy for risk.
    out.loc[out["scenario"] == "new_account", "account_segment"] = "new"
    out.loc[
        out["scenario"] == "high_value_legitimate", "account_segment"
    ] = "premium"
    out.loc[
        out["scenario"] == "shared_infrastructure", "account_segment"
    ] = "standard"

    return out


def attach_behavior_state(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Attach hidden simulation state used only to generate observable events."""
    out = accounts.copy()
    out["behavior_baseline"] = rng.lognormal(
        mean=np.log(140),
        sigma=0.45,
        size=len(out),
    )
    out["behavior_baseline"] = np.clip(out["behavior_baseline"], 35, 900)

    # High-value legitimate customers spend more, but remain low-risk.
    high_value = out["scenario"].eq("high_value_legitimate")
    out.loc[high_value, "behavior_baseline"] *= rng.uniform(
        2.5, 5.0, high_value.sum()
    )

    # Compromised accounts have a separate post-compromise regime.
    compromised = out["scenario"].eq("compromised")
    out["compromise_start_fraction"] = 0.62
    out["compromise_multiplier"] = 1.0
    out.loc[compromised, "compromise_multiplier"] = rng.uniform(
        4.0, 9.0, compromised.sum()
    )

    return out
