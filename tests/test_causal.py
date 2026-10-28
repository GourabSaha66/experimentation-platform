import numpy as np
import pandas as pd
import pytest

from expplat import causal
from expplat.common import UserError
from expplat.simulate import simulate_did, simulate_observational, simulate_synth


def test_psm_removes_confounding_bias():
    df = simulate_observational(n=4000, true_effect=2.0, seed=1)
    r = causal.analyze_psm(df, "treated", "outcome", ["age", "income", "prior_score"], n_boot=60, crosscheck=False)
    assert r["details"]["naive"] - 2.0 > 1.0                 # naive is clearly biased
    assert abs(r["details"]["att"] - 2.0) < 0.4              # matching is close to the truth
    assert max(abs(b["smd_after"]) for b in r["details"]["balance"]) < 0.1


def test_psm_crosschecks_agree_on_simulated_data():
    df = simulate_observational(n=3000, true_effect=2.0, seed=2)
    r = causal.analyze_psm(df, "treated", "outcome", ["age", "income", "prior_score"], n_boot=50, crosscheck=True)
    cc = r["details"]["crosscheck"]
    assert abs(cc["dowhy"]["att"] - 2.0) < 0.6
    assert cc["econml"]["available"] and abs(cc["econml"]["ate"] - 2.0) < 0.5


def test_psm_input_validation():
    df = simulate_observational(n=500, seed=1)
    with pytest.raises(UserError, match="0/1"):
        causal.analyze_psm(df.assign(treated=df.treated * 3), "treated", "outcome", ["age"], crosscheck=False)
    with pytest.raises(UserError, match="covariate"):
        causal.analyze_psm(df, "treated", "outcome", [], crosscheck=False)


def test_did_recovers_effect_and_flags_broken_parallel_trends():
    ests = [causal.analyze_did(simulate_did(effect=3.0, seed=s), "outcome", "group", "treated", "period", 7, unit_col="unit")
            for s in range(40, 60)]
    # unbiased on average, and the interval usually contains the truth
    assert abs(np.mean([e["details"]["estimate"] for e in ests]) - 3.0) < 0.25
    assert np.mean([e["details"]["ci_low"] < 3.0 < e["details"]["ci_high"] for e in ests]) >= 0.8
    good = ests[0]
    assert good["details"]["pretrend_p"] > 0.01
    bad = causal.analyze_did(simulate_did(effect=0.0, trend_gap=0.6, seed=4), "outcome", "group", "treated", "period", 7, unit_col="unit")
    assert bad["details"]["pretrend_p"] < 0.05
    assert any("trending differently" in w for w in bad["warnings"])


def test_did_placebo_false_positive_rate_is_low_without_effect():
    hits = 0
    for s in range(30):
        r = causal.analyze_did(simulate_did(effect=0.0, seed=100 + s), "outcome", "group", "treated", "period", 7, unit_col="unit")
        hits += r["details"]["placebo_p"] < 0.05
    assert hits <= 4


def test_synthetic_control_recovers_effect_and_weights_are_valid():
    df = simulate_synth(effect=-4.0, seed=3)
    r = causal.analyze_synth(df, "outcome", "unit", "period", "treated_unit", 21)
    assert abs(r["details"]["avg_post_gap"] + 4.0) < 0.6
    w = np.array(list(r["details"]["weights"].values()))
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0, abs=0.02)
    assert r["details"]["pre_rmspe"] < 0.5            # the blend tracks the treated unit before the change
    assert r["details"]["p_value"] < 0.1


def test_synthetic_control_no_effect_is_not_significant():
    df = simulate_synth(effect=0.0, seed=6)
    r = causal.analyze_synth(df, "outcome", "unit", "period", "treated_unit", 21)
    assert abs(r["details"]["avg_post_gap"]) < 0.8 and r["details"]["p_value"] > 0.1


def test_synthetic_control_errors():
    df = simulate_synth(seed=1)
    with pytest.raises(UserError, match="not in column"):
        causal.analyze_synth(df, "outcome", "unit", "period", "nobody", 21)
