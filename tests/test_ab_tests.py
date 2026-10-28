import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from expplat import ab_tests
from expplat.common import UserError
from expplat.simulate import simulate_ab, simulate_multi_metrics


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 2, 300), rng.normal(10.5, 4, 250)
    r = ab_tests.welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r["p_value"] == pytest.approx(ref.pvalue)
    assert r["ci_low"] < r["diff"] < r["ci_high"]
    assert r["diff"] == pytest.approx(b.mean() - a.mean())


def test_type_one_error_is_about_five_percent():
    rng = np.random.default_rng(7)
    hits = sum(ab_tests.welch_ttest(rng.normal(0, 1, 200), rng.normal(0, 1, 200))["p_value"] < 0.05 for _ in range(1000))
    assert 0.035 <= hits / 1000 <= 0.065


def test_confidence_interval_coverage():
    rng = np.random.default_rng(11)
    cover = 0
    for _ in range(600):
        r = ab_tests.welch_ttest(rng.normal(0, 1, 100), rng.normal(0.3, 1.5, 120))
        cover += r["ci_low"] <= 0.3 <= r["ci_high"]
    assert 0.92 <= cover / 600 <= 0.98


def test_two_proportions_matches_scipy_and_detects_effect():
    r = ab_tests.two_proportions(100, 1000, 150, 1000)
    chi2, p, *_ = stats.chi2_contingency(np.array([[100, 900], [150, 850]]), correction=False)
    assert r["p_value"] == pytest.approx(p)
    assert r["ci_low"] > 0 and r["diff"] == pytest.approx(0.05)


def test_srm_flags_broken_split():
    assert ab_tests.srm_check(5000, 5000)["mismatch"] is False
    assert ab_tests.srm_check(5000, 5400)["mismatch"] is True


def test_analyze_ttest_end_to_end_with_warning():
    df = simulate_ab(n=4000, lift=0.0, seed=3)
    df = pd.concat([df[df.group == "control"], df[df.group == "treatment"].head(1400)])
    r = ab_tests.analyze_ttest(df, "group", "revenue", "control", "treatment")
    assert any("Sample ratio mismatch" in w for w in r["warnings"])
    assert r["verdict"] in ("significant", "not_significant")
    assert "confidence interval" in r["conclusion"]


def test_chisq_binary_and_categorical():
    df = simulate_ab(n=20000, lift=0.02, seed=5)
    r = ab_tests.analyze_chisq(df, "group", "converted", "control", "treatment")
    assert r["verdict"] == "significant"
    assert r["details"]["diff"] > 0
    cat = pd.DataFrame({"group": ["control"] * 300 + ["treatment"] * 300,
                        "plan": ["free"] * 150 + ["pro"] * 100 + ["team"] * 50 + ["free"] * 100 + ["pro"] * 120 + ["team"] * 80})
    r2 = ab_tests.analyze_chisq(cat, "group", "plan", "control", "treatment")
    assert r2["details"]["dof"] == 2 and r2["details"]["cramers_v"] > 0


def test_unknown_group_label_is_a_clear_error():
    df = simulate_ab(n=200, seed=1)
    with pytest.raises(UserError, match="Could not find both groups"):
        ab_tests.analyze_ttest(df, "group", "revenue", "control", "variant_b")


def test_corrections_match_statsmodels_and_known_example():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    c = ab_tests.correct_pvalues(p, 0.05)
    assert c["bonferroni"]["p_adjusted"][0] == pytest.approx(0.01)
    assert sum(c["bonferroni"]["reject"]) == 1
    assert c["fdr_bh"]["p_adjusted"] == pytest.approx(multipletests(p, method="fdr_bh")[1].tolist())
    assert sum(c["fdr_bh"]["reject"]) >= sum(c["bonferroni"]["reject"])   # BH is never stricter


def test_bonferroni_controls_family_wise_error_and_bh_controls_fdr():
    rng = np.random.default_rng(21)
    fwer, fdp = [], []
    for _ in range(400):
        p = np.concatenate([rng.uniform(size=80), rng.beta(0.2, 8, 20)])   # 80 nulls, 20 real effects
        truth = np.array([False] * 80 + [True] * 20)
        bon = np.array(ab_tests.correct_pvalues(p, 0.05)["bonferroni"]["reject"])
        bh = np.array(ab_tests.correct_pvalues(p, 0.05)["fdr_bh"]["reject"])
        fwer.append((bon & ~truth).any())
        fdp.append((bh & ~truth).sum() / max(bh.sum(), 1))
    assert np.mean(fwer) <= 0.07
    assert np.mean(fdp) <= 0.07


def test_multiple_from_data_finds_real_metrics():
    df = simulate_multi_metrics(n=8000, effect=0.15, seed=4)
    r = ab_tests.analyze_multiple(df, "from_data", group_col="group", control="control", treatment="treatment",
                                  metric_cols=[c for c in df.columns if c.startswith("metric")])
    rows = {x["label"]: x for x in r["details"]["rows"]}
    assert rows["metric_01"]["sig_bh"]
    assert r["key_numbers"][0]["value"] == "12"
