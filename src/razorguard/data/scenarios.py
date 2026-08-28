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


SCENARIOS = np.array([
    "normal",
    "new_account",
    "high_value_legitimate",
    "compromised",
    "burst",
    "coordinated",
    "shared_infrastructure",
])


def assign_account_scenarios(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    config: ScenarioConfig | None = None,
) -> pd.DataFrame:
    config = config or ScenarioConfig()
    out = accounts.copy()
    out["scenario"] = rng.choice(
        SCENARIOS,
        size=len(out),
        p=config.probabilities,
    )

    out.loc[out["scenario"] == "new_account", "account_segment"] = "new"
    out.loc[
        out["scenario"] == "high_value_legitimate",
        "account_segment",
    ] = "premium"
    out.loc[
        out["scenario"] == "shared_infrastructure",
        "account_segment",
    ] = "standard"
    return out


def attach_behavior_state(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    end: pd.Timestamp = pd.Timestamp("2026-06-30 23:59:59"),
) -> pd.DataFrame:
    """Attach hidden account state used only by the simulation.

    Compromise timing is account-specific and hidden from model features.
    Activity intensity is also account-specific, allowing the generator to
    create coherent transaction histories instead of globally random rows.
    """
    out = accounts.copy()

    out["behavior_baseline"] = np.clip(
        rng.lognormal(
            mean=np.log(140),
            sigma=0.45,
            size=len(out),
        ),
        35,
        900,
    )

    high_value = out["scenario"].eq("high_value_legitimate")
    out.loc[high_value, "behavior_baseline"] *= rng.uniform(
        2.5,
        5.0,
        high_value.sum(),
    )

    compromised = out["scenario"].eq("compromised")
    span_seconds = max((end - start).total_seconds(), 1.0)
    offsets = rng.uniform(0.35, 0.75, len(out)) * span_seconds
    out["compromise_at"] = start + pd.to_timedelta(
        offsets,
        unit="s",
    )
    out.loc[~compromised, "compromise_at"] = pd.NaT

    out["compromise_multiplier"] = 1.0
    out.loc[compromised, "compromise_multiplier"] = rng.uniform(
        4.0,
        8.0,
        compromised.sum(),
    )

    out["secondary_device_id"] = [
        f"X{i:06d}" for i in range(len(out))
    ]

    # Account-specific event intensity in transactions/day. Overlap between
    # groups is intentional: scenario labels must not become trivial proxies.
    intensity = rng.lognormal(
        mean=np.log(1.7),
        sigma=0.55,
        size=len(out),
    )
    intensity = np.clip(intensity, 0.3, 12.0)

    new_mask = out["scenario"].eq("new_account").to_numpy()
    high_value_mask = out["scenario"].eq("high_value_legitimate").to_numpy()
    burst_mask = out["scenario"].eq("burst").to_numpy()
    coordinated_mask = out["scenario"].eq("coordinated").to_numpy()
    shared_mask = out["scenario"].eq("shared_infrastructure").to_numpy()

    intensity[new_mask] *= rng.uniform(0.55, 1.05, new_mask.sum())
    intensity[high_value_mask] *= rng.uniform(0.7, 1.25, high_value_mask.sum())
    intensity[burst_mask] *= rng.uniform(2.0, 3.5, burst_mask.sum())
    intensity[coordinated_mask] *= rng.uniform(1.4, 2.5, coordinated_mask.sum())
    intensity[shared_mask] *= rng.uniform(0.8, 1.15, shared_mask.sum())

    out["activity_rate_per_day"] = np.clip(
        intensity,
        0.2,
        30.0,
    )

    # A small probability of extra clustering is stored as hidden state. It is
    # translated into observable timestamps in the transaction generator.
    out["burst_probability"] = np.where(
        burst_mask,
        rng.uniform(0.45, 0.85, len(out)),
        rng.uniform(0.01, 0.08, len(out)),
    )

    return out
