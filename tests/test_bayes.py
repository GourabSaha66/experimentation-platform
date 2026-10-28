import numpy as np
import pytest
from scipy import stats

from expplat import bayes
from expplat.simulate import simulate_ab


def _exact_prob(sa, na, sb, nb, n=400_000, seed=1):
    rng = np.random.default_rng(seed)
    a = stats.beta(1 + sa, 1 + na - sa).rvs(n, random_state=rng)
    b = stats.beta(1 + sb, 1 + nb - sb).rvs(n, random_state=rng)
    return (b > a).mean()


def test_binary_matches_exact_beta_posterior():
    r = bayes.bayes_binary(195, 1971, 257, 2029, draws=3000, seed=1)
    assert r["prob_b_beats_a"] == pytest.approx(_exact_prob(195, 1971, 257, 2029), abs=0.01)
    assert r["rhat_max"] < 1.02


def test_no_difference_gives_middling_probability():
    r = bayes.bayes_binary(100, 1000, 100, 1000, draws=2000, seed=3)
    assert 0.3 < r["prob_b_beats_a"] < 0.7


def test_numeric_sufficient_statistics_equal_full_data_model():
    rng = np.random.default_rng(5)
    a, b = rng.normal(10, 3, 120), rng.normal(11, 3, 130)
    fast = bayes.bayes_numeric(a, b, draws=1500, seed=2)
    full = bayes.bayes_numeric(a, b, draws=1500, seed=2, full_data=True)
    assert fast["diff_mean"] == pytest.approx(full["diff_mean"], abs=0.1)
    assert fast["prob_b_beats_a"] == pytest.approx(full["prob_b_beats_a"], abs=0.04)


def test_numeric_credible_interval_covers_true_difference():
    rng = np.random.default_rng(8)
    a, b = rng.normal(10, 3, 400), rng.normal(11, 3, 400)
    r = bayes.bayes_numeric(a, b, draws=1500, seed=4)
    assert r["diff_low"] < 1.0 < r["diff_high"] and r["prob_b_beats_a"] > 0.95


def test_analyze_bayes_end_to_end():
    df = simulate_ab(n=6000, lift=0.03, seed=14)
    r = bayes.analyze_bayes(df, "group", "converted", "control", "treatment", draws=1000)
    assert r["verdict"] == "significant" and "probability" in r["conclusion"]
    r2 = bayes.analyze_bayes(df, "group", "revenue", "control", "treatment", metric_type="numeric", transform="log1p", draws=1000)
    assert r2["details"]["kind"] == "numeric"
