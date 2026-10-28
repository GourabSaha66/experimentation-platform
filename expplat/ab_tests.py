"""Classical fixed-horizon tests: Welch t-test, chi-square, and multiple-testing correction."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep

from .common import UserError, fmt, kn, numeric, pct, pfmt, require_columns, result, split_groups


# ---------------------------------------------------------------- helpers
def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 1.0) -> dict:
    """Sample ratio mismatch: did the groups come out the size they were planned to be?

    A tiny p-value means the split itself is off, which usually points to a bug in
    assignment or logging, and no test result can be trusted until it is explained.
    """
    total = n_control + n_treatment
    exp_c = total / (1 + expected_ratio)
    exp_t = total - exp_c
    chi2, p = stats.chisquare([n_control, n_treatment], [exp_c, exp_t])
    return {"chi2": float(chi2), "p_value": float(p), "mismatch": bool(p < 0.001),
            "n_control": int(n_control), "n_treatment": int(n_treatment)}


def _srm_warning(s: dict) -> list[str]:
    if s["mismatch"]:
        return [f"Sample ratio mismatch: the groups have {s['n_control']:,} and {s['n_treatment']:,} users, which is "
                f"very unlikely under the planned split ({pfmt(s['p_value'])}). Check how users were assigned and "
                "logged before trusting this result."]
    return []


def _hist_series(a: np.ndarray, b: np.ndarray, bins: int = 30) -> list[dict]:
    both = np.concatenate([a, b])
    lo, hi = np.percentile(both, [0.5, 99.5])
    if lo == hi:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    out = []
    for name, x in (("Control", a), ("Treatment", b)):
        dens, _ = np.histogram(np.clip(x, lo, hi), bins=edges, density=False)
        dens = dens / max(len(x), 1)
        out.append({"name": name, "fill": True, "points": [{"x": float(m), "y": float(d)} for m, d in zip(mid, dens)]})
    return out


# ---------------------------------------------------------------- Welch t-test
def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise UserError("Each group needs at least 2 rows with a value.")
    res = stats.ttest_ind(b, a, equal_var=False)
    ci = res.confidence_interval(confidence_level=1 - alpha)
    ma, mb = a.mean(), b.mean()
    diff = mb - ma
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    out = {
        "n_control": len(a), "n_treatment": len(b), "mean_control": float(ma), "mean_treatment": float(mb),
        "diff": float(diff), "ci_low": float(ci.low), "ci_high": float(ci.high),
        "t": float(res.statistic), "df": float(res.df), "p_value": float(res.pvalue),
        "cohens_d": float(diff / sp) if sp > 0 else float("nan"), "alpha": alpha,
    }
    if abs(ma) > 1e-12:
        # delta-method interval for the relative change b/a - 1 (approximate)
        se_a2, se_b2 = va / len(a), vb / len(b)
        ratio = mb / ma
        se_ratio = abs(ratio) * math.sqrt(se_b2 / mb ** 2 + se_a2 / ma ** 2) if mb != 0 else float("nan")
        z = stats.norm.ppf(1 - alpha / 2)
        out.update(rel_lift=float(ratio - 1), rel_low=float(ratio - 1 - z * se_ratio), rel_high=float(ratio - 1 + z * se_ratio))
    return out


def analyze_ttest(df: pd.DataFrame, group_col: str, metric_col: str, control: str, treatment: str,
                  alpha: float = 0.05, expected_ratio: float = 1.0) -> dict:
    require_columns(df, group_col, metric_col)
    ga, gb = split_groups(df, group_col, control, treatment)
    a = numeric(ga[metric_col], metric_col).dropna().to_numpy()
    b = numeric(gb[metric_col], metric_col).dropna().to_numpy()
    r = welch_ttest(a, b, alpha)
    srm = srm_check(len(a), len(b), expected_ratio)
    warnings = _srm_warning(srm)
    for name, x in (("control", a), ("treatment", b)):
        if len(x) > 20 and abs(stats.skew(x)) > 3:
            warnings.append(f"The {name} values are very skewed (long right tail). The t-test still works with many "
                            "users, but a few extreme values can dominate. Consider the Bayesian method with a log "
                            "transform, or check for outliers.")
            break
    if min(len(a), len(b)) < 30:
        warnings.append("Fewer than 30 users in a group: treat this result as rough.")

    sig = r["p_value"] < alpha
    direction = "higher" if r["diff"] > 0 else "lower"
    concl = (f"The treatment group's average is {fmt(r['mean_treatment'])} compared with {fmt(r['mean_control'])} in control, "
             f"a difference of {fmt(r['diff'])}")
    if "rel_lift" in r:
        concl += f" ({r['rel_lift'] * 100:+.1f}%)"
    concl += f". The {pct(1 - alpha, 0)} confidence interval for the difference runs from {fmt(r['ci_low'])} to {fmt(r['ci_high'])}. "
    if sig:
        concl += (f"This is statistically significant ({pfmt(r['p_value'])}): a gap this large would be unlikely if the "
                  f"treatment truly changed nothing. The treatment appears to make the metric {direction}.")
    else:
        concl += (f"This is not statistically significant ({pfmt(r['p_value'])}): the data are consistent with no real "
                  "difference. That is not proof of no effect. A small effect could still be hiding, especially with few users.")
    concl += " Statistical significance says an effect is unlikely to be chance, not that it is large enough to matter; judge that from the interval."

    key = [kn("Control mean", fmt(r["mean_control"]), f"n = {r['n_control']:,}"),
           kn("Treatment mean", fmt(r["mean_treatment"]), f"n = {r['n_treatment']:,}"),
           kn("Difference", fmt(r["diff"]), f"{pct(1 - alpha, 0)} CI {fmt(r['ci_low'])} to {fmt(r['ci_high'])}"),
           kn("p-value", f"{r['p_value']:.4f}" if r["p_value"] >= 0.0001 else "< 0.0001", "Welch t-test, two-sided"),
           kn("Effect size (Cohen's d)", fmt(r["cohens_d"]), "0.2 small, 0.5 medium, 0.8 large")]
    charts = [
        {"id": "dist", "title": "Distribution of the metric by group", "kind": "line", "x_label": metric_col,
         "y_label": "Share of group", "series": _hist_series(a, b)},
        {"id": "effect", "title": "Estimated difference (treatment minus control)", "kind": "forest",
         "rows": [{"label": "Difference", "estimate": r["diff"], "lo": r["ci_low"], "hi": r["ci_high"]}], "ref": 0},
    ]
    how = ("Welch's t-test compares the two group means without assuming the groups have equal variance. The p-value is "
           "the chance of seeing a gap at least this big if there were truly no difference. The interval is the range of "
           "differences the data are compatible with.")
    return result("ttest", "Two-group comparison of means (Welch t-test)", concl,
                  "significant" if sig else "not_significant", key, charts, warnings, {**r, "srm": srm}, how)


# ---------------------------------------------------------------- chi-square
def _binary_outcome(s: pd.Series, success_value: str | None) -> pd.Series | None:
    """Return a 0/1 series if the column is binary, else None."""
    vals = s.dropna().unique()
    if len(vals) > 2:
        return None
    sv = set(map(str, vals))
    if success_value is not None and str(success_value) in sv:
        return (s.astype(str) == str(success_value)).astype(float).where(s.notna())
    if sv <= {"0", "1", "0.0", "1.0"}:
        return pd.to_numeric(s, errors="coerce")
    low = {v.lower() for v in sv}
    if low <= {"true", "false"}:
        return s.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    if low <= {"yes", "no"}:
        return s.astype(str).str.lower().map({"yes": 1.0, "no": 0.0})
    if len(vals) == 2:
        raise UserError("The outcome column has two values, but I can't tell which one counts as a success. "
                        f"Values found: {', '.join(sorted(sv))}. Pick the success value in the settings.")
    return None


def two_proportions(sa: int, na: int, sb: int, nb: int, alpha: float = 0.05) -> dict:
    if na < 1 or nb < 1:
        raise UserError("Each group needs at least one user.")
    pa, pb = sa / na, sb / nb
    table = np.array([[sa, na - sa], [sb, nb - sb]])
    if table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
        chi2, p, dof, expected = float("nan"), 1.0, 1, table
        warn_degenerate = True
    else:
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        warn_degenerate = False
    lo, hi = confint_proportions_2indep(sb, nb, sa, na, method="newcomb", compare="diff", alpha=alpha)
    out = {"n_control": int(na), "n_treatment": int(nb), "successes_control": int(sa), "successes_treatment": int(sb),
           "rate_control": pa, "rate_treatment": pb, "diff": pb - pa, "ci_low": float(lo), "ci_high": float(hi),
           "chi2": float(chi2), "dof": int(dof), "p_value": float(p), "alpha": alpha, "min_expected": float(np.min(expected)),
           "degenerate": warn_degenerate}
    if pa > 0:
        out["rel_lift"] = pb / pa - 1
        # Newcombe interval for the ratio is not in statsmodels; use the log-ratio delta method
        if sa > 0 and sb > 0:
            se = math.sqrt(1 / sb - 1 / nb + 1 / sa - 1 / na)
            z = stats.norm.ppf(1 - alpha / 2)
            out["rel_low"] = math.exp(math.log(pb / pa) - z * se) - 1
            out["rel_high"] = math.exp(math.log(pb / pa) + z * se) - 1
    if out["min_expected"] < 5:
        _, pf = stats.fisher_exact(table)
        out["fisher_p"] = float(pf)
    return out


def analyze_chisq(df: pd.DataFrame, group_col: str, outcome_col: str, control: str, treatment: str,
                  alpha: float = 0.05, success_value: str | None = None, expected_ratio: float = 1.0) -> dict:
    require_columns(df, group_col, outcome_col)
    ga, gb = split_groups(df, group_col, control, treatment)
    both = pd.concat([ga, gb])
    outcome = _binary_outcome(both[outcome_col], success_value)
    warnings: list[str] = []

    if outcome is not None:
        oa = outcome.loc[ga.index].dropna()
        ob = outcome.loc[gb.index].dropna()
        r = two_proportions(int(oa.sum()), len(oa), int(ob.sum()), len(ob), alpha)
        srm = srm_check(len(oa), len(ob), expected_ratio)
        warnings += _srm_warning(srm)
        p_used = r.get("fisher_p", r["p_value"]) if r["min_expected"] < 5 else r["p_value"]
        if r["min_expected"] < 5:
            warnings.append("Some expected counts are below 5, so the chi-square approximation is weak. The p-value shown "
                            "is Fisher's exact test instead.")
        if r["degenerate"]:
            warnings.append("One outcome never happens in the data, so there is nothing to compare yet.")
        sig = p_used < alpha
        concl = (f"{pct(r['rate_treatment'], 2)} of the treatment group had the outcome ({r['successes_treatment']:,} of "
                 f"{r['n_treatment']:,}) compared with {pct(r['rate_control'], 2)} in control ({r['successes_control']:,} of "
                 f"{r['n_control']:,}). That is a difference of {r['diff'] * 100:+.2f} percentage points "
                 f"({pct(1 - alpha, 0)} interval {r['ci_low'] * 100:+.2f} to {r['ci_high'] * 100:+.2f}). ")
        if sig:
            concl += (f"The difference is statistically significant ({pfmt(p_used)}), so it is unlikely to be chance alone. "
                      f"The treatment appears to {'raise' if r['diff'] > 0 else 'lower'} the rate.")
        else:
            concl += (f"The difference is not statistically significant ({pfmt(p_used)}). The data are consistent with no "
                      "real change, but a small change cannot be ruled out.")
        key = [kn("Control rate", pct(r["rate_control"], 2), f"{r['successes_control']:,} / {r['n_control']:,}"),
               kn("Treatment rate", pct(r["rate_treatment"], 2), f"{r['successes_treatment']:,} / {r['n_treatment']:,}"),
               kn("Difference", f"{r['diff'] * 100:+.2f} pp", f"{pct(1 - alpha, 0)} CI {r['ci_low'] * 100:+.2f} to {r['ci_high'] * 100:+.2f}"),
               kn("p-value", f"{p_used:.4f}" if p_used >= 0.0001 else "< 0.0001", "chi-square" if r["min_expected"] >= 5 else "Fisher exact")]
        if "rel_lift" in r:
            rel_note = (f"CI {r['rel_low'] * 100:+.1f}% to {r['rel_high'] * 100:+.1f}%" if "rel_low" in r else "")
            key.append(kn("Relative change", f"{r['rel_lift'] * 100:+.1f}%", rel_note))
        charts = [
            {"id": "rates", "title": "Rate by group (with confidence interval of the difference below)", "kind": "bar",
             "categories": ["Control", "Treatment"], "series": [{"name": "Rate", "values": [r["rate_control"] * 100, r["rate_treatment"] * 100]}],
             "y_label": "% with the outcome"},
            {"id": "effect", "title": "Difference in rate (percentage points)", "kind": "forest",
             "rows": [{"label": "Treatment minus control", "estimate": r["diff"] * 100, "lo": r["ci_low"] * 100, "hi": r["ci_high"] * 100}],
             "ref": 0},
        ]
        details = {**r, "srm": srm, "p_value_used": p_used}
    else:
        # general contingency table over all groups in the file for the two chosen arms
        tab = pd.crosstab(both[group_col].astype(str), both[outcome_col].astype(str))
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            raise UserError("Need at least two groups and two outcome categories for a chi-square test.")
        chi2, p, dof, expected = stats.chi2_contingency(tab.to_numpy())
        n = tab.to_numpy().sum()
        v = math.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        if expected.min() < 5:
            warnings.append("Some expected counts are below 5, so the chi-square approximation is weak.")
        sig = p < alpha
        concl = (f"The outcome categories are distributed differently across the groups "
                 f"(chi-square = {chi2:.1f}, {dof} degrees of freedom, {pfmt(p)}), with an association strength (Cramér's V) "
                 f"of {v:.2f}. " if sig else
                 f"There is no clear evidence that the outcome categories differ between the groups "
                 f"(chi-square = {chi2:.1f}, {dof} degrees of freedom, {pfmt(p)}). ")
        concl += "A significant result says the groups differ somewhere in the table, not which category drives it. Look at the chart."
        props = tab.div(tab.sum(axis=1), axis=0) * 100
        key = [kn("Chi-square", f"{chi2:.2f}", f"{dof} degrees of freedom"), kn("p-value", f"{p:.4f}" if p >= 0.0001 else "< 0.0001"),
               kn("Cramér's V", f"{v:.2f}", "0.1 small, 0.3 medium, 0.5 large")]
        charts = [{"id": "mix", "title": "Outcome mix by group (%)", "kind": "bar", "categories": list(props.index),
                   "series": [{"name": str(c), "values": props[c].tolist()} for c in props.columns], "stacked": True,
                   "y_label": "% of group"}]
        details = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": v, "table": tab.to_dict()}

    how = ("The chi-square test checks whether the outcome is spread across groups the way it would be if group did not "
           "matter. For a yes/no outcome the interval for the difference uses Newcombe's method, which behaves well even "
           "for small or extreme rates.")
    return result("chisq", "Comparison of rates (chi-square test)", concl, "significant" if sig else "not_significant",
                  key, charts, warnings, details, how)


# ---------------------------------------------------------------- multiple testing
def correct_pvalues(pvals: list[float], alpha: float = 0.05) -> dict:
    p = np.asarray(pvals, float)
    if len(p) < 2:
        raise UserError("Give at least two p-values to correct.")
    out = {}
    for key, label in (("bonferroni", "Bonferroni"), ("holm", "Holm"), ("fdr_bh", "Benjamini-Hochberg")):
        rej, padj, _, _ = multipletests(p, alpha=alpha, method=key)
        out[key] = {"label": label, "reject": rej.tolist(), "p_adjusted": padj.tolist()}
    return out


def analyze_multiple(df: pd.DataFrame, mode: str, alpha: float = 0.05, group_col: str | None = None,
                     control: str | None = None, treatment: str | None = None, metric_cols: list[str] | None = None,
                     label_col: str | None = None, p_col: str | None = None) -> dict:
    if mode == "from_data":
        if not (group_col and control is not None and treatment is not None and metric_cols):
            raise UserError("Choose the group column, the two groups, and at least two metric columns.")
        require_columns(df, group_col, *metric_cols)
        ga, gb = split_groups(df, group_col, control, treatment)
        rows = []
        for m in metric_cols:
            r = welch_ttest(numeric(ga[m], m).dropna().to_numpy(), numeric(gb[m], m).dropna().to_numpy(), alpha)
            rows.append({"label": m, "p_value": r["p_value"], "diff": r["diff"]})
    elif mode == "p_values":
        if not p_col:
            raise UserError("Choose the column that holds the p-values.")
        require_columns(df, p_col, label_col)
        p = numeric(df[p_col], p_col)
        labels = df[label_col].astype(str) if label_col else pd.Series([f"Test {i + 1}" for i in range(len(df))], index=df.index)
        rows = [{"label": l, "p_value": float(v), "diff": None} for l, v in zip(labels, p) if not math.isnan(v)]
        if any(not 0 <= r["p_value"] <= 1 for r in rows):
            raise UserError("p-values must be between 0 and 1.")
    else:
        raise UserError("Mode must be 'from_data' or 'p_values'.")
    if len(rows) < 2:
        raise UserError("You need at least two tests to correct for multiple testing.")
    corr = correct_pvalues([r["p_value"] for r in rows], alpha)
    for i, r in enumerate(rows):
        r["bonferroni_p"] = corr["bonferroni"]["p_adjusted"][i]
        r["holm_p"] = corr["holm"]["p_adjusted"][i]
        r["bh_p"] = corr["fdr_bh"]["p_adjusted"][i]
        r["sig_raw"] = r["p_value"] < alpha
        r["sig_bonferroni"] = bool(corr["bonferroni"]["reject"][i])
        r["sig_holm"] = bool(corr["holm"]["reject"][i])
        r["sig_bh"] = bool(corr["fdr_bh"]["reject"][i])
    k = len(rows)
    n_raw = sum(r["sig_raw"] for r in rows)
    n_bon = sum(r["sig_bonferroni"] for r in rows)
    n_bh = sum(r["sig_bh"] for r in rows)
    expected_fp = k * alpha
    concl = (f"You ran {k} tests. Without any correction, {n_raw} come out significant at {pct(alpha, 0)}, but by chance alone you "
             f"would expect about {expected_fp:.1f} false alarms among {k} tests even if nothing were real. "
             f"After correction, {n_bon} remain significant with Bonferroni (strict: keeps the chance of ANY false alarm at "
             f"{pct(alpha, 0)}) and {n_bh} with Benjamini-Hochberg (keeps the share of false discoveries among your "
             f"discoveries at about {pct(alpha, 0)}). ")
    if n_raw > n_bh:
        concl += "The tests that disappear after correction should be treated as leads to retest, not as wins."
    elif n_raw == n_bon:
        concl += "The results are the same with and without correction, so they look solid."
    key = [kn("Tests run", str(k)), kn("Significant, uncorrected", str(n_raw)), kn("Significant, Bonferroni", str(n_bon)),
           kn("Significant, Benjamini-Hochberg", str(n_bh)), kn("Expected false alarms (uncorrected)", f"{expected_fp:.1f}")]
    rows_sorted = sorted(rows, key=lambda r: r["p_value"])
    charts = [{"id": "adj", "title": "Raw and adjusted p-values (dashed line: significance level)", "kind": "bar",
               "categories": [r["label"] for r in rows_sorted],
               "series": [{"name": "Raw", "values": [r["p_value"] for r in rows_sorted]},
                          {"name": "Bonferroni", "values": [min(r["bonferroni_p"], 1) for r in rows_sorted]},
                          {"name": "Benjamini-Hochberg", "values": [r["bh_p"] for r in rows_sorted]}],
               "hline": alpha, "y_label": "p-value"}]
    how = ("Every extra test is another chance for a false alarm. Bonferroni multiplies each p-value by the number of tests "
           "(very safe, loses power). Holm is a uniformly stronger version of the same guarantee. Benjamini-Hochberg controls "
           "the false discovery rate instead, which is usually the better trade-off when you are screening many metrics or variants.")
    return result("multiple", "Multiple-testing correction", concl, "info", key, charts, [], {"rows": rows_sorted, "alpha": alpha}, how)
