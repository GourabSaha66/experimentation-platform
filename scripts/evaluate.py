"""Measure how well each method actually works, using simulations with known answers and real data sets.

    python scripts/evaluate.py            # full run (about 10-15 minutes)
    python scripts/evaluate.py --quick    # a fifth of the repetitions, for a fast check

Writes docs/evaluation.json (all numbers) and docs/evaluation.md (readable tables).
Every random draw uses a fixed seed, so the numbers can be reproduced.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from expplat import ab_tests, bayes, causal, power, sequential  # noqa: E402
from expplat.simulate import simulate_ab, simulate_did, simulate_observational, simulate_synth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ARGS = ap.parse_args()
SCALE = 0.2 if ARGS.quick else 1.0


def reps(n: int) -> int:
    return max(20, int(n * SCALE))


OUT: dict = {"quick": ARGS.quick}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ 1. t-test and the power calculator
def eval_ttest_and_power():
    rng = np.random.default_rng(1)
    S = reps(4000)
    n = 400
    a = rng.normal(0, 1, (S, n)); b = rng.normal(0, 1, (S, n))
    p_null = stats.ttest_ind(b, a, equal_var=False, axis=1).pvalue
    # calculator sample size for d = 0.2 at 80% power
    n_calc = power.plan("means", "sample_size", sd=1.0, mde=0.2)["details"]["n_control"]
    a = rng.normal(0, 1, (S, n_calc)); b = rng.normal(0.2, 1, (S, n_calc))
    p_alt = stats.ttest_ind(b, a, equal_var=False, axis=1).pvalue
    # confidence interval coverage
    cover = 0
    C = reps(1500)
    for _ in range(C):
        r = ab_tests.welch_ttest(rng.normal(0, 1, 120), rng.normal(0.3, 1.6, 90))
        cover += r["ci_low"] <= 0.3 <= r["ci_high"]
    # proportions: calculator vs simulated chi-square power
    n_p = power.plan("proportions", "sample_size", baseline_rate=0.10, mde=0.10, mde_type="relative")["details"]["n_control"]
    S2 = reps(2000)
    sa = rng.binomial(n_p, 0.10, S2); sb = rng.binomial(n_p, 0.11, S2)
    pw = 0
    for x, y in zip(sa, sb):
        t = np.array([[x, n_p - x], [y, n_p - y]])
        pw += stats.chi2_contingency(t, correction=False)[1] < 0.05
    sa0 = rng.binomial(n_p, 0.10, S2); sb0 = rng.binomial(n_p, 0.10, S2)
    fp = sum(stats.chi2_contingency(np.array([[x, n_p - x], [y, n_p - y]]), correction=False)[1] < 0.05 for x, y in zip(sa0, sb0))
    return {
        "ttest_false_positive_rate": float((p_null < 0.05).mean()), "ttest_sims": S,
        "calculator_n_per_group_d0.2": int(n_calc), "ttest_empirical_power_at_calculator_n": float((p_alt < 0.05).mean()), "target_power": 0.8,
        "ci95_coverage": cover / C, "ci_sims": C,
        "proportions_n_per_group_10_to_11pct": int(n_p), "chisq_empirical_power_at_calculator_n": pw / S2, "chisq_false_positive_rate": fp / S2, "chisq_sims": S2,
    }


# ------------------------------------------------------------------ 2. peeking
def eval_sequential():
    rng = np.random.default_rng(2026)
    S = reps(2000)
    n, looks = 2000, 10
    naive = seq = 0
    for _ in range(S):
        x = rng.normal(0, 1, n); t = rng.random(n) < 0.5
        path = sequential.msprt_path(x, t, tau=0.1, n_looks=looks, min_per_group=50)
        naive += (path["p_naive"] <= 0.05).any()
        seq += (path["p_always_valid"] <= 0.05).any()
    # a real effect: how often does each approach detect it, and how early does the sequential test stop?
    S2 = reps(600)
    n2, eff = 4000, 0.09
    fixed_hit = seq_hit = 0
    stops = []
    for _ in range(S2):
        a = rng.normal(0, 1, n2 // 2); b = rng.normal(eff, 1, n2 // 2)
        fixed_hit += stats.ttest_ind(b, a, equal_var=False).pvalue < 0.05
        x = np.empty(n2); t = np.zeros(n2, bool)
        t[1::2] = True; x[~t] = a; x[t] = b
        path = sequential.msprt_path(x, t, tau=0.09, n_looks=60)
        hit = path[path["p_always_valid"] <= 0.05]
        if len(hit):
            seq_hit += 1; stops.append(int(hit.iloc[0]["n_total"]))
    return {"aa_trials": S, "aa_users": n, "aa_looks": looks, "aa_false_positive_naive_peeking": naive / S, "aa_false_positive_sequential": seq / S,
            "effect_trials": S2, "effect_users": n2, "effect_size_sd": eff, "power_fixed_single_look": fixed_hit / S2, "power_sequential": seq_hit / S2,
            "median_stop_users_when_detected": float(np.median(stops)) if stops else None,
            "share_stopped_before_end": float(np.mean([s < n2 for s in stops])) if stops else None}


# ------------------------------------------------------------------ 3. multiple testing
def eval_multiple():
    rng = np.random.default_rng(21)
    S = reps(1000)
    m, m_real = 100, 20
    res = {k: {"any_false": [], "fdp": [], "power": []} for k in ("uncorrected", "bonferroni", "holm", "bh")}
    for _ in range(S):
        z_null = rng.normal(0, 1, m - m_real); z_real = rng.normal(3.0, 1, m_real)
        p = 2 * (1 - stats.norm.cdf(np.abs(np.concatenate([z_null, z_real]))))
        truth = np.array([False] * (m - m_real) + [True] * m_real)
        c = ab_tests.correct_pvalues(p.tolist(), 0.05)
        decisions = {"uncorrected": p < 0.05, "bonferroni": np.array(c["bonferroni"]["reject"]), "holm": np.array(c["holm"]["reject"]),
                     "bh": np.array(c["fdr_bh"]["reject"])}
        for k, rej in decisions.items():
            res[k]["any_false"].append(bool((rej & ~truth).any()))
            res[k]["fdp"].append((rej & ~truth).sum() / max(rej.sum(), 1))
            res[k]["power"].append((rej & truth).sum() / m_real)
    return {"simulations": S, "hypotheses": m, "truly_different": m_real,
            **{k: {"family_wise_error_rate": float(np.mean(v["any_false"])), "false_discovery_rate": float(np.mean(v["fdp"])), "power": float(np.mean(v["power"]))}
               for k, v in res.items()}}


# ------------------------------------------------------------------ 4. Bayesian
def eval_bayes():
    cases = [(100, 1000, 100, 1000), (195, 1971, 257, 2029), (30, 300, 45, 300), (500, 5000, 560, 5000), (8, 80, 15, 80), (1000, 10000, 1090, 10000)]
    rows = []
    for i, (sa, na, sb, nb) in enumerate(cases):
        t = time.time()
        r = bayes.bayes_binary(sa, na, sb, nb, draws=2000, seed=i)
        el = time.time() - t
        rng = np.random.default_rng(100 + i)
        ex = (stats.beta(1 + sb, 1 + nb - sb).rvs(400000, random_state=rng) > stats.beta(1 + sa, 1 + na - sa).rvs(400000, random_state=rng)).mean()
        rows.append({"case": f"{sa}/{na} vs {sb}/{nb}", "pymc_prob_b_beats_a": r["prob_b_beats_a"], "exact_prob": float(ex),
                     "abs_difference": abs(r["prob_b_beats_a"] - float(ex)), "rhat": r["rhat_max"], "ess_min": r["ess_min"], "seconds": el})
    rng = np.random.default_rng(5)
    a, b = rng.normal(10, 3, 120), rng.normal(11, 3, 130)
    t = time.time(); fast = bayes.bayes_numeric(a, b, draws=1500, seed=2); t_fast = time.time() - t
    t = time.time(); full = bayes.bayes_numeric(a, b, draws=1500, seed=2, full_data=True); t_full = time.time() - t
    # calibration: when the model says "B beats A with probability p", how often is that true? (simulation, small)
    K = reps(120)
    rng = np.random.default_rng(77)
    cover = 0
    for k in range(K):
        pa, pb = rng.beta(2, 18), rng.beta(2, 18)  # truth drawn from the prior family the model assumes broadly
        sa, sb = rng.binomial(400, pa), rng.binomial(400, pb)
        r = bayes.bayes_binary(int(sa), 400, int(sb), 400, prior_a=2, prior_b=18, draws=800, tune=500, seed=k)
        cover += r["diff_low"] <= (pb - pa) <= r["diff_high"]
    return {"binary_vs_exact": rows, "max_abs_difference": max(r["abs_difference"] for r in rows),
            "numeric_fast_vs_full": {"diff_mean_fast": fast["diff_mean"], "diff_mean_full": full["diff_mean"],
                                      "prob_fast": fast["prob_b_beats_a"], "prob_full": full["prob_b_beats_a"], "seconds_fast": t_fast, "seconds_full": t_full},
            "credible_interval_coverage": cover / K, "coverage_runs": K}


# ------------------------------------------------------------------ 5. causal
def eval_causal():
    out: dict = {}
    # LaLonde: the randomised result is the benchmark
    nsw = pd.read_csv(ROOT / "sample_data" / "lalonde_nsw_experiment.csv")
    exp = float(nsw[nsw.treat == 1].re78.mean() - nsw[nsw.treat == 0].re78.mean())
    d = pd.read_csv(ROOT / "sample_data" / "lalonde_nsw_cps.csv")
    covs = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
    t = time.time()
    r = causal.analyze_psm(d, "treat", "re78", covs, n_boot=reps(300), refute=True, known_effect=exp)
    dt = r["details"]
    out["lalonde"] = {"experimental_benchmark": exp, "naive_cps_comparison": dt["naive"], "psm_att": dt["att"], "psm_ci": [dt["ci_low"], dt["ci_high"]],
                      "dowhy_psm": dt["crosscheck"]["dowhy"].get("att"), "dowhy_placebo": dt["crosscheck"]["dowhy"].get("refutation"),
                      "econml_dr": dt["crosscheck"]["econml"], "treated_matched": f"{dt['n_matched']} of {dt['n_treated']}",
                      "worst_balance_after": max(abs(b["smd_after"]) for b in dt["balance"]),
                      "worst_balance_before": max(abs(b["smd_before"]) for b in dt["balance"]), "seconds": time.time() - t}
    # simulated observational: bias before and after, many seeds
    K = reps(40)
    naive, att, dwy, ecm = [], [], [], []
    for s in range(K):
        df = simulate_observational(n=3000, true_effect=2.0, seed=500 + s)
        r = causal.analyze_psm(df, "treated", "outcome", ["age", "income", "prior_score"], n_boot=50, crosscheck=(s < 8))
        naive.append(r["details"]["naive"]); att.append(r["details"]["att"])
        if s < 8:
            dwy.append(r["details"]["crosscheck"]["dowhy"]["att"]); ecm.append(r["details"]["crosscheck"]["econml"]["ate"])
    out["simulated_psm"] = {"true_effect": 2.0, "runs": K, "naive_mean": float(np.mean(naive)), "psm_mean": float(np.mean(att)), "psm_rmse": float(np.sqrt(np.mean((np.array(att) - 2) ** 2))),
                            "dowhy_mean_first8": float(np.mean(dwy)), "econml_mean_first8": float(np.mean(ecm))}
    # difference-in-differences
    K = reps(60)
    est, cov, plc_hits, pre_hits = [], [], 0, 0
    for s in range(K):
        r = causal.analyze_did(simulate_did(effect=3.0, seed=900 + s), "outcome", "group", "treated", "period", 7, unit_col="unit")
        est.append(r["details"]["estimate"]); cov.append(r["details"]["ci_low"] < 3.0 < r["details"]["ci_high"])
        pre_hits += r["details"]["pretrend_p"] < 0.05
    null_hits = 0
    for s in range(K):
        r = causal.analyze_did(simulate_did(effect=0.0, seed=1900 + s), "outcome", "group", "treated", "period", 7, unit_col="unit")
        null_hits += r["details"]["placebo_p"] < 0.05
    drift_hits = 0
    for s in range(K):
        r = causal.analyze_did(simulate_did(effect=0.0, trend_gap=0.5, seed=2900 + s), "outcome", "group", "treated", "period", 7, unit_col="unit")
        drift_hits += r["details"]["pretrend_p"] < 0.05
    out["did"] = {"true_effect": 3.0, "runs": K, "mean_estimate": float(np.mean(est)), "rmse": float(np.sqrt(np.mean((np.array(est) - 3) ** 2))),
                  "ci_coverage_clustered": float(np.mean(cov)), "false_positive_rate_no_effect_placebo": null_hits / K,
                  "pretrend_flag_rate_when_parallel": pre_hits / K, "pretrend_flag_rate_when_diverging": drift_hits / K}
    ob = pd.read_csv(ROOT / "sample_data" / "organ_donations_did.csv")
    r = causal.analyze_did(ob, "Rate", "group", "California", "Quarter_Num", 4, unit_col="State")
    out["organ_donations"] = {k: r["details"][k] for k in ("estimate", "ci_low", "ci_high", "p_value", "placebo_p", "pretrend_p")}
    # synthetic control
    K = reps(30)
    gaps, ps0 = [], 0
    for s in range(K):
        r = causal.analyze_synth(simulate_synth(effect=-4.0, seed=300 + s), "outcome", "unit", "period", "treated_unit", 21)
        gaps.append(r["details"]["avg_post_gap"])
    for s in range(K):
        r = causal.analyze_synth(simulate_synth(effect=0.0, seed=700 + s), "outcome", "unit", "period", "treated_unit", 21)
        ps0 += r["details"]["p_value"] <= 0.1
    out["synthetic_control"] = {"true_effect": -4.0, "runs": K, "mean_estimate": float(np.mean(gaps)), "rmse": float(np.sqrt(np.mean((np.array(gaps) + 4) ** 2))),
                                "false_positive_rate_at_p0.1_no_effect": ps0 / K}
    tx = pd.read_csv(ROOT / "sample_data" / "texas_prison_synth.csv")
    r = causal.analyze_synth(tx, "bmprison", "state", "year", "Texas", 1993)
    out["texas"] = {k: r["details"][k] for k in ("avg_post_gap", "relative_gap_pct", "p_value", "pre_rmspe", "ratio")}
    return out


# ------------------------------------------------------------------ 6. speed through the real API
def eval_api_speed():
    from fastapi.testclient import TestClient
    from expplat.main import app
    from expplat.samples import EXAMPLES
    client = TestClient(app)
    rows = []
    for ex in EXAMPLES:
        raw = (ROOT / "sample_data" / ex["file"]).read_bytes()
        n_rows = raw.count(b"\n") - 1
        params = dict(ex["params"])
        if ex["method"] == "psm":
            params["refute"] = False
        t = time.time()
        r = client.post(f"/api/analyze/{ex['method']}", files={"file": (ex["file"], raw, "text/csv")}, data={"params": json.dumps(params)})
        rows.append({"example": ex["id"], "method": ex["method"], "rows": n_rows, "status": r.status_code, "seconds": round(time.time() - t, 2)})
    t = time.time()
    for _ in range(20):
        client.post("/api/power", json={"kind": "proportions", "solve_for": "sample_size", "baseline_rate": 0.1, "mde": 0.1})
    return {"analyses": rows, "power_endpoint_ms": (time.time() - t) / 20 * 1000}


def main():
    steps = [("ttest_and_power", eval_ttest_and_power), ("sequential", eval_sequential), ("multiple_testing", eval_multiple),
             ("bayesian", eval_bayes), ("causal", eval_causal), ("api_speed", eval_api_speed)]
    for name, fn in steps:
        log(f"running {name} ...")
        OUT[name] = fn()
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "evaluation.json").write_text(json.dumps(OUT, indent=2, default=float))
    (ROOT / "docs" / "evaluation.md").write_text(render(OUT))
    log("done. wrote docs/evaluation.json and docs/evaluation.md")


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def render(o: dict) -> str:
    t, s, m, b, c, a = o["ttest_and_power"], o["sequential"], o["multiple_testing"], o["bayesian"], o["causal"], o["api_speed"]
    L = ["# Evaluation results", "",
         "Generated by `python scripts/evaluate.py" + (" --quick" if o["quick"] else "") + "`. All simulations use fixed seeds.", "",
         "## 1. Classical tests and the power calculator", "",
         "| Check | Result | Target |", "|---|---|---|",
         f"| t-test false-positive rate when there is no effect ({t['ttest_sims']:,} simulated experiments) | {pc(t['ttest_false_positive_rate'], 2)} | 5% |",
         f"| t-test power at the calculator's sample size ({t['calculator_n_per_group_d0.2']} per group, effect 0.2 sd) | {pc(t['ttest_empirical_power_at_calculator_n'])} | 80% |",
         f"| 95% confidence-interval coverage ({t['ci_sims']:,} simulations) | {pc(t['ci95_coverage'])} | 95% |",
         f"| Conversion test power at the calculator's size ({t['proportions_n_per_group_10_to_11pct']:,} per group, 10% to 11%) | {pc(t['chisq_empirical_power_at_calculator_n'])} | 80% |",
         f"| Chi-square false-positive rate when there is no effect | {pc(t['chisq_false_positive_rate'], 2)} | 5% |", "",
         "## 2. Sequential testing (peeking)", "",
         "| Check | Result |", "|---|---|",
         f"| False alarms with {s['aa_looks']} peeks at an A/A test, ordinary p-value ({s['aa_trials']:,} trials) | **{pc(s['aa_false_positive_naive_peeking'])}** |",
         f"| False alarms, same data, sequential test | **{pc(s['aa_false_positive_sequential'])}** |",
         f"| Power with a real effect ({s['effect_size_sd']} sd, {s['effect_users']:,} users): one look at the end | {pc(s['power_fixed_single_look'])} |",
         f"| Power, sequential test | {pc(s['power_sequential'])} |",
         f"| Median users at which the sequential test stopped (when it detected the effect) | {s['median_stop_users_when_detected']:,.0f} of {s['effect_users']:,} |", "",
         "## 3. Multiple-testing correction", "",
         f"{m['hypotheses']} hypotheses, {m['truly_different']} of them truly different, {m['simulations']:,} simulated experiments.", "",
         "| Method | Chance of any false positive | False discovery rate | Share of real effects found |", "|---|---|---|---|"]
    for k, n in (("uncorrected", "No correction"), ("bonferroni", "Bonferroni"), ("holm", "Holm"), ("bh", "Benjamini-Hochberg")):
        L.append(f"| {n} | {pc(m[k]['family_wise_error_rate'])} | {pc(m[k]['false_discovery_rate'])} | {pc(m[k]['power'])} |")
    L += ["", "## 4. Bayesian A/B analysis (PyMC)", "", "| Data (control vs treatment) | PyMC P(B beats A) | Exact (Beta posterior) | Difference | R-hat | Seconds |", "|---|---|---|---|---|---|"]
    for r in b["binary_vs_exact"]:
        L.append(f"| {r['case']} | {r['pymc_prob_b_beats_a']:.4f} | {r['exact_prob']:.4f} | {r['abs_difference']:.4f} | {r['rhat']:.3f} | {r['seconds']:.1f} |")
    nf = b["numeric_fast_vs_full"]
    L += ["", f"- Largest gap between PyMC and the exact answer: **{b['max_abs_difference']:.4f}**.",
          f"- Numeric metric, sufficient-statistics model vs full-data model: difference in means {nf['diff_mean_fast']:.3f} vs {nf['diff_mean_full']:.3f}; P(B>A) {nf['prob_fast']:.3f} vs {nf['prob_full']:.3f}; {nf['seconds_fast']:.1f}s vs {nf['seconds_full']:.1f}s.",
          f"- 95% credible-interval coverage of the true difference over {b['coverage_runs']} simulated experiments: **{pc(b['credible_interval_coverage'])}**.", "",
          "## 5. Observational causal methods", "", "### LaLonde job training (real data)", "",
          "Treated group: 185 men from the randomised programme. Comparison group: 15,992 men from a national survey (CPS). The randomised experiment gives the benchmark answer.", "",
          "| Estimate | Value ($ of 1978 earnings) |", "|---|---|"]
    ll = c["lalonde"]
    L.append(f"| Randomised experiment (benchmark) | {ll['experimental_benchmark']:,.0f} |")
    L.append(f"| Naive comparison with survey controls | {ll['naive_cps_comparison']:,.0f} |")
    L.append(f"| Propensity score matching (this tool), 95% interval {ll['psm_ci'][0]:,.0f} to {ll['psm_ci'][1]:,.0f} | {ll['psm_att']:,.0f} |")
    if ll["dowhy_psm"] is not None:
        L.append(f"| DoWhy matching | {ll['dowhy_psm']:,.0f} |")
    ec = ll["econml_dr"]
    if ec.get("available"):
        L.append(f"| EconML doubly robust (overlap region, n={ec['n_used']:,}), 95% interval {ec['ci_low']:,.0f} to {ec['ci_high']:,.0f} | {ec['ate']:,.0f} |")
    rf = ll["dowhy_placebo"]
    L += ["", f"- Treated units matched: {ll['treated_matched']}. Largest covariate imbalance: {ll['worst_balance_before']:.2f} before, {ll['worst_balance_after']:.2f} after matching."]
    if rf:
        L.append(f"- DoWhy placebo-treatment refutation: placebo effect {rf['placebo_effect']:.1f}, p = {rf['p_value']:.2f}.")
    sp = c["simulated_psm"]
    L += ["", f"### Simulated confounded data (true effect {sp['true_effect']}, {sp['runs']} runs)", "",
          f"- Naive difference averaged **{sp['naive_mean']:.2f}** (biased). Matching averaged **{sp['psm_mean']:.2f}** (RMSE {sp['psm_rmse']:.2f}).",
          f"- DoWhy matching averaged {sp['dowhy_mean_first8']:.2f} and EconML doubly robust {sp['econml_mean_first8']:.2f} over the first 8 runs.", "",
          "### Difference-in-differences", ""]
    d = c["did"]
    L += [f"- Simulated panel, true effect {d['true_effect']} ({d['runs']} runs): mean estimate **{d['mean_estimate']:.2f}**, RMSE {d['rmse']:.2f}, clustered 95% interval covered the truth {pc(d['ci_coverage_clustered'])} of the time.",
          f"- No real effect: placebo p-value below 0.05 in {pc(d['false_positive_rate_no_effect_placebo'])} of runs.",
          f"- Pre-trend test flags a problem in {pc(d['pretrend_flag_rate_when_parallel'])} of runs with parallel trends and {pc(d['pretrend_flag_rate_when_diverging'])} of runs where the groups were drifting apart."]
    od = c["organ_donations"]
    L += [f"- California organ donations (real data): effect {od['estimate']:.4f} ({od['ci_low']:.4f} to {od['ci_high']:.4f}); clustered p = {od['p_value']:.4f}, but placebo p = {od['placebo_p']:.3f} because only one state was treated.", "", "### Synthetic control", ""]
    sc = c["synthetic_control"]
    tx = c["texas"]
    L += [f"- Simulated panel, true effect {sc['true_effect']} ({sc['runs']} runs): mean estimate **{sc['mean_estimate']:.2f}**, RMSE {sc['rmse']:.2f}.",
          f"- No real effect: placebo p <= 0.1 in {pc(sc['false_positive_rate_at_p0.1_no_effect'])} of runs.",
          f"- Texas prison expansion (real data): average gap {tx['avg_post_gap']:,.0f} ({tx['relative_gap_pct']:+.0f}%), post/pre error ratio {tx['ratio']:.1f}, placebo p = {tx['p_value']:.3f}, pre-period fit RMSPE {tx['pre_rmspe']:,.0f}.", "",
          "## 6. Speed through the API", "", "| Example | Method | Rows | HTTP | Seconds |", "|---|---|---|---|---|"]
    for r in a["analyses"]:
        L.append(f"| {r['example']} | {r['method']} | {r['rows']:,} | {r['status']} | {r['seconds']} |")
    L.append(f"\nPower calculator endpoint: {a['power_endpoint_ms']:.0f} ms per request.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
