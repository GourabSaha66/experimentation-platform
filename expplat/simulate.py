"""Data simulators. Every one takes a seed, so any result can be reproduced exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_ab(n: int = 20000, base_rate: float = 0.10, lift: float = 0.0, mean_revenue: float = 25.0,
                revenue_sigma: float = 0.9, revenue_effect: float = 0.0, share_treatment: float = 0.5,
                seed: int = 0, days: int = 28) -> pd.DataFrame:
    """A two-arm web experiment. ``lift`` is the ABSOLUTE change in conversion rate in the treatment arm.

    Users arrive over ``days`` days (timestamps are sorted), converters spend a log-normal amount.
    """
    rng = np.random.default_rng(seed)
    treat = rng.random(n) < share_treatment
    rate = np.where(treat, base_rate + lift, base_rate)
    converted = (rng.random(n) < rate).astype(int)
    mu = np.log(mean_revenue) - revenue_sigma ** 2 / 2
    rev = np.where(converted == 1, rng.lognormal(mu, revenue_sigma, n) * (1 + np.where(treat, revenue_effect, 0.0)), 0.0)
    start = pd.Timestamp("2026-09-01")
    ts = start + pd.to_timedelta(np.sort(rng.random(n)) * days * 86400, unit="s")
    return pd.DataFrame({"user_id": np.arange(1, n + 1), "timestamp": ts.floor("s"),
                         "group": np.where(treat, "treatment", "control"), "converted": converted,
                         "revenue": np.round(rev, 2)})


def simulate_multi_metrics(n: int = 6000, n_metrics: int = 12, n_real: int = 3, effect: float = 0.12, seed: int = 0) -> pd.DataFrame:
    """One experiment, many metrics. The first ``n_real`` metrics have a true effect (in standard deviations)."""
    rng = np.random.default_rng(seed)
    treat = rng.random(n) < 0.5
    cols = {"group": np.where(treat, "treatment", "control")}
    for j in range(n_metrics):
        shift = effect if j < n_real else 0.0
        cols[f"metric_{j + 1:02d}"] = np.round(rng.normal(0, 1, n) + shift * treat, 4)
    return pd.DataFrame(cols)


def simulate_observational(n: int = 4000, true_effect: float = 2.0, confounding: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Treatment is more likely for some people, and those same people have higher outcomes anyway.

    A naive comparison therefore overstates the effect; adjusting for the covariates recovers ``true_effect``.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    income = rng.normal(50, 15, n)
    prior = rng.normal(0, 1, n)
    z = confounding * (0.04 * (age - 40) + 0.03 * (income - 50) + 0.6 * prior) - 0.2
    treated = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    y = 5 + 0.1 * (age - 40) + 0.08 * (income - 50) + 1.5 * prior + true_effect * treated + rng.normal(0, 2, n)
    return pd.DataFrame({"age": age.round(1), "income": income.round(1), "prior_score": prior.round(3), "treated": treated, "outcome": y.round(3)})


def simulate_did(n_units: int = 40, n_treated: int = 8, n_periods: int = 12, treat_period: int = 7, effect: float = 3.0,
                 trend_gap: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Panel with unit and time effects. ``trend_gap`` makes the treated group drift away (breaks parallel trends)."""
    rng = np.random.default_rng(seed)
    rows = []
    unit_fx = rng.normal(10, 3, n_units)
    time_fx = np.cumsum(rng.normal(0.3, 0.3, n_periods))
    for u in range(n_units):
        tr = u < n_treated
        for t in range(1, n_periods + 1):
            y = unit_fx[u] + time_fx[t - 1] + rng.normal(0, 1)
            if tr:
                y += trend_gap * t + (effect if t >= treat_period else 0.0)
            rows.append({"unit": f"unit_{u + 1:02d}", "period": t, "group": "treated" if tr else "control", "outcome": round(y, 3)})
    return pd.DataFrame(rows)


def simulate_synth(n_donors: int = 25, n_periods: int = 30, treat_period: int = 21, effect: float = -4.0, seed: int = 0) -> pd.DataFrame:
    """Treated unit is a true mixture of three donors plus noise; the effect starts at ``treat_period``."""
    rng = np.random.default_rng(seed)
    factors = np.cumsum(rng.normal(0, 1, (n_periods, 3)), axis=0)
    load = rng.normal(1, 0.5, (n_donors, 3))
    donors = factors @ load.T + rng.normal(0, 0.3, (n_periods, n_donors)) + rng.normal(20, 5, n_donors)
    w = np.zeros(n_donors)
    w[:3] = [0.5, 0.3, 0.2]
    treated = donors @ w + rng.normal(0, 0.2, n_periods)
    treated[treat_period - 1:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(n_periods):
            rows.append({"unit": f"donor_{j + 1:02d}", "period": t + 1, "outcome": round(float(donors[t, j]), 3)})
    for t in range(n_periods):
        rows.append({"unit": "treated_unit", "period": t + 1, "outcome": round(float(treated[t]), 3)})
    return pd.DataFrame(rows)
