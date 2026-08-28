from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def generate_account_event_stream(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    n_transactions: int,
    start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    end: pd.Timestamp = pd.Timestamp("2026-06-30 23:59:59"),
) -> pd.DataFrame:
    """
    Generate transaction events from account-level activity intensity.

    The total number of events is exactly n_transactions.

    Compromised accounts have:
      - a normal pre-compromise regime
      - an elevated post-compromise regime

    The hidden scenario/state is only used by the simulator and is never
    exposed as a model feature.
    """

    accounts = accounts.reset_index(drop=True).copy()

    horizon_days = max(
        (end - start).total_seconds() / 86_400.0,
        1e-9,
    )

    rates = accounts["activity_rate_per_day"].to_numpy(dtype=float)
    rates = np.clip(rates, 0.1, 30.0)

    scenarios = accounts["scenario"].to_numpy()
    compromised_mask = scenarios == "compromised"

    compromise_at = accounts["compromise_at"]

    pre_days = np.full(len(accounts), horizon_days)

    valid_compromise = compromised_mask & compromise_at.notna().to_numpy()

    if valid_compromise.any():
        pre_days[valid_compromise] = np.clip(
            (
                compromise_at[valid_compromise]
                - start
            ).dt.total_seconds().to_numpy()
            / 86_400.0,
            1.0,
            horizon_days - 1.0,
        )

    post_days = horizon_days - pre_days

    # Compromised accounts become more active after compromise.
    post_rates = rates.copy()
    post_rates[compromised_mask] *= 2.25

    expected_pre = rates * pre_days
    expected_post = post_rates * post_days

    expected_total = expected_pre + expected_post

    # Scale expected counts to the requested dataset size.
    scale = n_transactions / max(
        expected_total.sum(),
        1e-9,
    )

    expected_pre *= scale
    expected_post *= scale

    # Poisson draws provide natural account-level count variation.
    pre_counts = rng.poisson(expected_pre)
    post_counts = np.zeros(len(accounts), dtype=int)
    post_counts[compromised_mask] = rng.poisson(
        expected_post[compromised_mask]
    )

    # Correct the stochastic draw so that we produce exactly
    # n_transactions.
    current_total = int(
        pre_counts.sum() + post_counts.sum()
    )

    diff = n_transactions - current_total

    if diff > 0:
        weights = expected_total / expected_total.sum()
        additions = rng.multinomial(
            diff,
            weights,
        )

        # Prefer the post-compromise regime for compromised accounts
        # only through their already increased expected intensity.
        pre_counts += additions

    elif diff < 0:

        combined = np.concatenate(
            [pre_counts, post_counts]
        )

        removable_indices = np.flatnonzero(
            combined > 0
        )

        chosen = rng.choice(
            removable_indices,
            size=min(
                -diff,
                len(removable_indices),
            ),
            replace=False,
        )

        combined[chosen] -= 1

        pre_counts = combined[: len(accounts)]
        post_counts = combined[len(accounts):]

    rows: list[dict] = []

    for idx, account in accounts.iterrows():

        account_id = account["account_id"]

        n_pre = int(pre_counts[idx])
        n_post = int(post_counts[idx])

        if n_pre > 0:

            pre_end = (
                pd.Timestamp(account["compromise_at"])
                if compromised_mask[idx]
                else end
            )

            pre_span = max(
                (pre_end - start).total_seconds(),
                1.0,
            )

            timestamps = (
                start
                + pd.to_timedelta(
                    rng.uniform(
                        0,
                        pre_span,
                        n_pre,
                    ),
                    unit="s",
                )
            )

            rows.extend(
                {
                    "account_id": account_id,
                    "timestamp": ts,
                    "regime": (
                        "pre"
                        if compromised_mask[idx]
                        else "baseline"
                    ),
                }
                for ts in timestamps
            )

        if n_post > 0:

            post_start = pd.Timestamp(
                account["compromise_at"]
            )

            post_span = max(
                (end - post_start).total_seconds(),
                1.0,
            )

            timestamps = (
                post_start
                + pd.to_timedelta(
                    rng.uniform(
                        0,
                        post_span,
                        n_post,
                    ),
                    unit="s",
                )
            )

            rows.extend(
                {
                    "account_id": account_id,
                    "timestamp": ts,
                    "regime": "post",
                }
                for ts in timestamps
            )

    events = pd.DataFrame(rows)

    # Defensive invariant.
    if len(events) != n_transactions:
        raise RuntimeError(
            f"Event generator produced {len(events)} "
            f"events; expected {n_transactions}"
        )

    return (
        events
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )