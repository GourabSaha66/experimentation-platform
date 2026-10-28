import numpy as np
import pytest

from expplat import sequential
from expplat.common import UserError
from expplat.simulate import simulate_ab


def _aa_trial(rng, n=2000, looks=10):
    x = rng.normal(0, 1, n)
    t = rng.random(n) < 0.5
    path = sequential.msprt_path(x, t, tau=0.1, n_looks=looks, min_per_group=50)
    return (path["p_naive"] <= 0.05).any(), (path["p_always_valid"] <= 0.05).any()


def test_peeking_inflates_ordinary_p_values_but_not_sequential():
    rng = np.random.default_rng(2026)
    res = [_aa_trial(rng) for _ in range(1500)]
    naive = np.mean([a for a, _ in res])
    seq = np.mean([b for _, b in res])
    assert naive > 0.12          # ten peeks at an A/A test: roughly 20% false alarms
    assert seq < 0.05            # the sequential test keeps it below alpha


def test_stops_early_on_a_real_effect_and_interval_covers_truth():
    df = simulate_ab(n=40000, base_rate=0.10, lift=0.02, seed=9)
    r = sequential.analyze_sequential(df, "group", "converted", "control", "treatment", order_col="timestamp")
    assert r["verdict"] == "significant"
    assert r["details"]["stopped_at"] < 40000
    last = r["details"]["looks"][-1]
    assert last["cs_low"] < 0.02 < last["cs_high"]


def test_no_effect_stays_inconclusive():
    df = simulate_ab(n=6000, lift=0.0, seed=12)
    r = sequential.analyze_sequential(df, "group", "converted", "control", "treatment", order_col="timestamp")
    assert r["verdict"] == "inconclusive"


def test_confidence_sequence_is_nested_and_p_values_monotone():
    rng = np.random.default_rng(3)
    path = sequential.msprt_path(rng.normal(0, 1, 3000), rng.random(3000) < 0.5, 0.1)
    assert (np.diff(path["p_always_valid"]) <= 1e-12).all()
    assert (np.diff(path["cs_low"]) >= -1e-12).all() and (np.diff(path["cs_high"]) <= 1e-12).all()


def test_too_little_data_is_a_user_error():
    with pytest.raises(UserError):
        sequential.msprt_path(np.arange(10.0), np.arange(10) % 2 == 0, 0.1)
