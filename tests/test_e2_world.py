import numpy as np
import pandas as pd

from razorguard.data.scenarios import ScenarioConfig, assign_account_scenarios, attach_behavior_state
from razorguard.data.validation import validate_account_scenarios, validate_transactions


def _accounts(n=2000):
    rng = np.random.default_rng(42)
    created = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 420, n), unit="D"
    )
    accounts = pd.DataFrame(
        {
            "account_id": [f"A{i:06d}" for i in range(n)],
            "created_at": created,
            "home_country": rng.choice(["IN", "US"], n),
            "device_id": [f"D{i:06d}" for i in rng.integers(0, int(n * .8), n)],
            "account_segment": rng.choice(["new", "standard", "premium"], n),
        }
    )
    accounts = assign_account_scenarios(rng, accounts, ScenarioConfig())
    return attach_behavior_state(rng, accounts)


def test_e2_scenarios_cover_all_archetypes():
    accounts = _accounts()
    expected = {
        "normal",
        "new_account",
        "high_value_legitimate",
        "compromised",
        "burst",
        "coordinated",
        "shared_infrastructure",
    }
    assert set(accounts["scenario"].unique()) == expected


def test_e2_behavior_state_is_positive():
    accounts = _accounts()
    assert (accounts["behavior_baseline"] > 0).all()
    assert (accounts["compromise_multiplier"] > 0).all()


def test_account_scenario_validation():
    accounts = _accounts()
    validate_account_scenarios(accounts)


def test_transaction_validation_requires_chronological_order():
    tx = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "account_id": ["a", "a"],
            "merchant_id": ["m", "m"],
            "device_id": ["d", "d"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "amount": [100.0, 200.0],
            "is_chargeback": [0, 1],
        }
    )
    validate_transactions(tx)
