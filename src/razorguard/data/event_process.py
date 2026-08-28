from __future__ import annotations

import numpy as np
import pandas as pd


def _split_interval_counts(
    rng: np.random.Generator,
    rates_per_day: np.ndarray,
    days: float,
    target_total: int,
) -> np.ndarray:
    """Allocate an exact total event count proportional to account intensity."""
    weights = np.maximum(rates_per_day, 1e-9)
    expected = days * weights
    expected *= target_total / expected.sum()

    counts = np.floor(expected).astype(int)
    remainder = target_total - int(counts.sum())
    if remainder > 0:
        fractional = expected - counts
        selected = np.argsort(fractional)[-remainder:]
        counts[selected] += 1
    elif remainder < 0:
        fractional = expected - counts
        candidates = np.argsort(fractional)[: -remainder]
        for idx in candidates:
            if counts[idx] > 0:
                counts[idx] -= 1

    # Shuffle equal-probability ties so account IDs do not dictate allocation.
    order = rng.permutation(len(counts))
    return counts[order][np.argsort(order)]


def generate_account_event_stream(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    n_transactions: int,
    start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    end: pd.Timestamp = pd.Timestamp("2026-06-30 23:59:59"),
) -> pd.DataFrame:
    """Generate an exact-size event stream from account-level intensities.

    Events are generated in two regimes for compromised accounts so the
    post-compromise activity increase is observable from history alone.
    """
    total_seconds = (end - start).total_seconds()
    horizon_days = total_seconds / 86_400.0

    scenario = accounts["scenario"].to_numpy()
    baseline_rate = accounts["activity_rate_per_day"].to_numpy(dtype=float)
    compromise_at = accounts["compromise_at"].to_numpy()

    normal_counts = np.zeros(len(accounts), dtype=int)
    post_counts = np.zeros(len(accounts), dtype=int)

    compromised = scenario == "compromised"
    normal_fraction = np.full(len(accounts), 1.0)
    normal_fraction[compromised] = np.clip(
        (compromise_at[compromised] - np.datetime64(start))
        / np.timedelta64(1, "D")
        / horizon_days,
        0.20,
        0.90,
    )

    pre_days = horizon_days * normal_fraction
    post_days = horizon_days - pre_days

    # Compromised accounts accelerate after the hidden transition; other
    # scenarios keep their baseline intensity.
    post_rate = baseline_rate.copy()
    post_rate[compromised] *= 2.2

    expected_pre = baseline_rate * pre_days
    expected_post = post_rate * post_days
    expected_total = expected_pre + expected_post

    scale = n_transactions / max(expected_total.sum(), 1e-9)
    expected_pre *= scale
    expected_post *= scale

    normal_counts = rng.poisson(expected_pre)
    post_counts = rng.poisson(expected_post)

    # Correct stochastic sampling to hit the requested dataset size exactly.
    current_total = int(normal_counts.sum() + post_counts.sum())
    diff = n_transactions - current_total

    if diff > 0:
        weights = expected_total / expected_total.sum()
        additions = rng.multinomial(diff, weights)
        normal_counts += additions
    elif diff < 0:
        removable = np.concatenate([normal_counts, post_counts])
        for flat_idx in rng.choice(
            len(removable), size=min(-diff, int(removable.sum())), replace=False
        ):
            if removable[flat_idx] > 0:
                removable[flat_idx] -= 1
        normal_counts = removable[: len(accounts)]
        post_counts = removable[len(accounts) :]

    rows: list[dict] = []

    for i, account in accounts.iterrows():
        account_id = account["account_id"]
        base_rate = float(account["activity_rate_per_day"])

        n_pre = int(normal_counts[i])
        n_post = int(post_counts[i]) if compromised[i] else 0

        if n_pre:
            pre_end = account["compromise_at"] if compromised[i] else end
            pre_end = pd.Timestamp(pre_end)
            pre_span = max((pre_end - start).total_seconds(), 1.0)
            offsets = rng.uniform(0, pre_span, n_pre)
            for offset in offsets:
                rows.append(
                    {
                        "account_id": account_id,
                        "timestamp": start + pd.to_timedelta(offset, unit="s"),
                        "regime": "pre" if compromised[i] else "baseline",
                    }
                )

        if n_post:
            post_start = pd.Timestamp(account["compromise_at"])
            post_span = max((end - post_start).total_seconds(), 1.0)
            offsets = rng.uniform(0, post_span, n_post)
            for offset in offsets:
                rows.append(
                    {
                        "account_id": account_id,
                        "timestamp": post_start + pd.to_timedelta(offset, unit="s"),
                        "regime": "post",
                    }
                )

    events = pd.DataFrame(rows)
    if len(events) != n_transactions:
        # Deterministic fallback for the tiny residual caused by integer
        # allocation: add/remove uniformly sampled events without changing
        # account-level semantics materially.
        diff = n_transactions - len(events)
        if diff > 0:
            extras = rng.choice(len(accounts), size=diff, replace=True)
            extra_rows = []
            for idx in extras:
                extra_rows.append(
                    {
                        "account_id": accounts.iloc[idx]["account_id"],
                        "timestamp": start + pd.to_timedelta(
                            rng.uniform(0, total_seconds), unit="s"
                        ),
                        "regime": "baseline",
                    }
                )
            events = pd.concat([events, pd.DataFrame(extra_rows)], ignore_index=True)
        elif diff < 0:
            events = events.iloc[:n_transactions].copy()

    return events.sort_values("timestamp", kind="stable").reset_index(drop=True)
