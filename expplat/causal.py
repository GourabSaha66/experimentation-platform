"""Observational causal methods: propensity score matching, difference-in-differences, synthetic control.

Use these when you cannot randomise. They give an answer only under assumptions that the data
cannot prove (no unmeasured confounding, parallel trends, a good synthetic match). Every
result below therefore comes with diagnostics and plain-English warnings.

DoWhy is used to state the causal model, identify the effect and refute the estimate;
EconML supplies the doubly robust estimator that is used as a cross-check.
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .common import UserError, fmt, kn, numeric, pct, pfmt, require_columns, result

for _name in ("dowhy", "econml"):
    logging.getLogger(_name).setLevel(logging.ERROR)


# =====================================================================  propensity scores
def _prep(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    require_columns(df, treatment, outcome, *covariates)
    if not covariates:
        raise UserError("Choose at least one covariate (a pre-treatment characteristic that affects both who is treated and the outcome).")
    if treatment in covariates or outcome in covariates:
        raise UserError("The treatment and outcome columns cannot also be covariates.")
    d = df[[treatment, outcome] + covariates].copy()
    t = d[treatment]
    if t.dtype == bool:
        d[treatment] = t.astype(int)
    else:
        d[treatment] = pd.to_numeric(t, errors="coerce")
    vals = set(d[treatment].dropna().unique())
    if not vals <= {0, 1}:
        raise UserError(f"The treatment column '{treatment}' must be 0/1 (1 = treated). Found values: {sorted(map(str, vals))[:6]}.")
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    cov_used = []
    for c in covariates:
        if pd.api.types.is_numeric_dtype(d[c]) or d[c].dtype == bool:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
            cov_used.append(c)
        else:
            dummies = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
            d = pd.concat([d.drop(columns=[c]), dummies], axis=1)
            cov_used += list(dummies.columns)
    before = len(d)
    d = d.dropna()
    if before - len(d) > 0.2 * before:
        raise UserError(f"{before - len(d):,} of {before:,} rows have missing values in the chosen columns. Clean the data or pick other columns.")
    d[treatment] = d[treatment].astype(int)
    if d[treatment].sum() < 10 or (1 - d[treatment]).sum() < 10:
        raise UserError("Need at least 10 treated and 10 untreated rows.")
    return d.reset_index(drop=True), cov_used


def propensity(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(Xs, t)
    return np.clip(lr.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)


def _match_att(ps: np.ndarray, t: np.ndarray, y: np.ndarray, caliper_sd: float = 0.2):
    """1:1 nearest-neighbour matching on the logit of the propensity score, with replacement."""
    logit = np.log(ps / (1 - ps))
    caliper = caliper_sd * logit.std()
    tr, co = np.where(t == 1)[0], np.where(t == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(logit[co].reshape(-1, 1))
    dist, idx = nn.kneighbors(logit[tr].reshape(-1, 1))
    dist, idx = dist.ravel(), co[idx.ravel()]
    ok = dist <= caliper
    if ok.sum() == 0:
        return float("nan"), tr[ok], idx[ok], caliper
    att = float((y[tr[ok]] - y[idx[ok]]).mean())
    return att, tr[ok], idx[ok], caliper


def _smd(x: np.ndarray, t: np.ndarray, w_c: np.ndarray | None = None, rows_t: np.ndarray | None = None,
         rows_c: np.ndarray | None = None) -> float:
    pooled = math.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    if rows_t is None:
        return float((x[t == 1].mean() - x[t == 0].mean()) / pooled)
    return float((x[rows_t].mean() - x[rows_c].mean()) / pooled)


def _dowhy_psm(d: pd.DataFrame, treatment: str, outcome: str, covs: list[str], refute: bool, n_sims: int, seed: int) -> dict:
    from dowhy import CausalModel  # imported lazily: heavy
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CausalModel(data=d[[treatment, outcome] + covs], treatment=treatment, outcome=outcome, common_causes=covs)
        ident = model.identify_effect(proceed_when_unidentifiable=True)
        est = model.estimate_effect(ident, method_name="backdoor.propensity_score_matching", target_units="att")
        out = {"att": float(est.value), "refutation": None}
        if refute:
            ref = model.refute_estimate(ident, est, method_name="placebo_treatment_refuter", placebo_type="permute",
                                        num_simulations=n_sims)
            p = ref.refutation_result.get("p_value") if getattr(ref, "refutation_result", None) else None
            out["refutation"] = {"name": "Placebo treatment (shuffled treatment labels)", "placebo_effect": float(ref.new_effect),
                                 "p_value": float(p) if p is not None else None}
    return out


def _econml_dr(d: pd.DataFrame, treatment: str, outcome: str, covs: list[str], ps: np.ndarray, trim: tuple[float, float], seed: int) -> dict:
    from econml.dr import LinearDRLearner  # lazy
    from sklearn.ensemble import GradientBoostingRegressor
    keep = (ps >= trim[0]) & (ps <= trim[1])
    t = d[treatment].to_numpy()
    if t[keep].sum() < 10 or (1 - t[keep]).sum() < 10:
        return {"available": False, "reason": "Too few treated or untreated units remain after dropping units with extreme propensity scores."}
    W = StandardScaler().fit_transform(d[covs].to_numpy())[keep]
    y = d[outcome].to_numpy()[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = LinearDRLearner(model_propensity=LogisticRegression(C=1e6, max_iter=5000),
                              model_regression=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
                              cv=3, min_propensity=0.02, random_state=seed)
        est.fit(y, t[keep], X=None, W=W)
        inf = est.ate_inference()
        lo, hi = inf.conf_int_mean(alpha=0.05)
    return {"available": True, "ate": float(np.ravel(inf.mean_point)[0]), "ci_low": float(np.ravel(lo)[0]), "ci_high": float(np.ravel(hi)[0]),
            "n_used": int(keep.sum()), "n_treated_used": int(t[keep].sum()), "trim": list(trim)}


def analyze_psm(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str], caliper: float = 0.2,
                n_boot: int = 200, crosscheck: bool = True, refute: bool = False, known_effect: float | None = None,
                alpha: float = 0.05, seed: int = 0) -> dict:
    d, covs = _prep(df, treatment, outcome, covariates)
    t, y, X = d[treatment].to_numpy(), d[outcome].to_numpy(float), d[covs].to_numpy(float)
    ps = propensity(X, t)
    att, rt, rc, cal = _match_att(ps, t, y, caliper)
    n_t = int(t.sum())
    if not rt.size:
        raise UserError("No treated unit has a similar untreated unit. The groups barely overlap, so matching cannot work with these covariates.")
    naive = float(y[t == 1].mean() - y[t == 0].mean())

    # bootstrap interval (re-estimates the propensity model each time)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(d)
    for _ in range(max(50, int(n_boot))):
        ix = rng.integers(0, n, n)
        tb = t[ix]
        if tb.sum() < 5 or (1 - tb).sum() < 5:
            continue
        pb = propensity(X[ix], tb)
        a, *_ = _match_att(pb, tb, y[ix], caliper)
        if not math.isnan(a):
            boots.append(a)
    lo, hi = (np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(boots) > 20 else (float("nan"), float("nan")))
    se = float(np.std(boots, ddof=1)) if len(boots) > 20 else float("nan")

    # covariate balance
    rows = []
    for j, c in enumerate(covs):
        x = X[:, j]
        before = _smd(x, t)
        after = _smd(x, t, rows_t=rt, rows_c=rc)
        rows.append({"covariate": c, "smd_before": before, "smd_after": after})
    worst_after = max(abs(r["smd_after"]) for r in rows)
    dropped = n_t - len(rt)

    warnings_: list[str] = []
    if dropped > 0.2 * n_t:
        warnings_.append(f"{dropped} of {n_t} treated units ({pct(dropped / n_t, 0)}) had no close match and were dropped, so the result "
                         "describes the treated units that can be matched, not all of them.")
    if worst_after > 0.1:
        bad = [r["covariate"] for r in rows if abs(r["smd_after"]) > 0.1]
        warnings_.append("Some covariates are still imbalanced after matching (standardised difference above 0.1): "
                         + ", ".join(bad) + ". The estimate may still be biased by them.")
    warnings_.append("Matching only balances what you measured. If an unmeasured factor drives both treatment and outcome, the "
                     "estimate is biased and no diagnostic here can show it.")
    if len(rc) and len(set(rc)) < 0.3 * len(rc):
        warnings_.append("Many treated units were matched to the same few untreated units, so the estimate rests on a small part of the comparison group.")

    # cross-checks with DoWhy and EconML
    cross: dict = {}
    if crosscheck:
        try:
            cross["dowhy"] = _dowhy_psm(d, treatment, outcome, covs, refute, 20, seed)
        except Exception as e:  # keep the main result if a cross-check fails
            cross["dowhy"] = {"error": str(e)[:200]}
        try:
            cross["econml"] = _econml_dr(d, treatment, outcome, covs, ps, (0.05, 0.95), seed)
        except Exception as e:
            cross["econml"] = {"available": False, "reason": str(e)[:200]}

    ci_txt = f"{pct(1 - alpha, 0)} bootstrap interval {fmt(lo)} to {fmt(hi)}"
    sig = not (lo <= 0 <= hi)
    concl = (f"After matching each treated unit to a similar untreated unit, the treated group's outcome is {fmt(att)} higher than "
             f"its matched comparison group ({ci_txt}) " if att >= 0 else
             f"After matching each treated unit to a similar untreated unit, the treated group's outcome is {fmt(abs(att))} lower than "
             f"its matched comparison group ({ci_txt}) ")
    concl += (f"- this is the estimated effect on the treated units. A plain comparison of averages without matching would have given "
              f"{fmt(naive)}, so adjusting for the covariates changed the answer by {fmt(att - naive)}. ")
    concl += ("The interval excludes zero, so the effect looks real if the covariates capture everything that drives both treatment and the outcome. "
              if sig else "The interval includes zero, so the data do not clearly show an effect. ")
    if known_effect is not None:
        concl += f"For reference, the known true effect for this data set is {fmt(known_effect)}. "
    dw = cross.get("dowhy", {})
    ec = cross.get("econml", {})
    if "att" in dw:
        concl += f"DoWhy's own matching estimate is {fmt(dw['att'])}. "
    if ec.get("available"):
        concl += (f"EconML's doubly robust estimate on the region where treated and untreated units overlap is {fmt(ec['ate'])} "
                  f"({fmt(ec['ci_low'])} to {fmt(ec['ci_high'])}). ")
    concl += "Agreement between methods builds confidence; disagreement means the answer depends on modelling choices."

    key = [kn("Effect on the treated (matched)", fmt(att), ci_txt),
           kn("Naive difference (no adjustment)", fmt(naive), "treated minus untreated averages"),
           kn("Treated units matched", f"{len(rt):,} of {n_t:,}", f"caliper {caliper} sd of the logit score"),
           kn("Worst covariate imbalance after", f"{worst_after:.3f}", "aim for below 0.1")]
    if "att" in dw:
        key.append(kn("DoWhy matching estimate", fmt(dw["att"]), "DoWhy backdoor matching"))
    if dw.get("refutation"):
        rf = dw["refutation"]
        key.append(kn("DoWhy placebo refutation", f"effect {fmt(rf['placebo_effect'])}", f"p = {rf['p_value']:.2f}; high p = passes" if rf["p_value"] is not None else ""))
    if ec.get("available"):
        key.append(kn("EconML doubly robust", fmt(ec["ate"]), f"{fmt(ec['ci_low'])} to {fmt(ec['ci_high'])}; n = {ec['n_used']:,}"))

    forest = [{"label": "Naive difference", "estimate": naive, "lo": None, "hi": None},
              {"label": "Propensity matching (this tool)", "estimate": att, "lo": float(lo), "hi": float(hi)}]
    if "att" in dw:
        forest.append({"label": "DoWhy matching", "estimate": dw["att"], "lo": None, "hi": None})
    if ec.get("available"):
        forest.append({"label": "EconML doubly robust", "estimate": ec["ate"], "lo": ec["ci_low"], "hi": ec["ci_high"]})
    charts = [
        {"id": "estimates", "title": "Effect estimates compared", "kind": "forest", "rows": forest, "ref": known_effect if known_effect is not None else 0,
         "ref_label": "known true effect" if known_effect is not None else "no effect"},
        {"id": "balance", "title": "Covariate balance (absolute standardised difference)", "kind": "bar", "horizontal": True,
         "categories": [r["covariate"] for r in rows],
         "series": [{"name": "Before matching", "values": [abs(r["smd_before"]) for r in rows]},
                    {"name": "After matching", "values": [abs(r["smd_after"]) for r in rows]}], "hline": 0.1,
         "y_label": "|standardised difference|"},
    ]
    edges = np.linspace(0, 1, 21)
    mid = (edges[:-1] + edges[1:]) / 2
    sh = []
    for name, m in (("Treated", t == 1), ("Untreated", t == 0)):
        h, _ = np.histogram(ps[m], bins=edges)
        sh.append({"name": name, "fill": True, "points": [{"x": float(a), "y": float(b / max(m.sum(), 1))} for a, b in zip(mid, h)]})
    charts.append({"id": "overlap", "title": "Overlap: propensity score of treated and untreated units", "kind": "line",
                   "x_label": "Propensity score (chance of being treated given the covariates)", "y_label": "Share of group", "series": sh})
    how = ("Propensity score matching estimates each unit's chance of being treated from its characteristics, then pairs every treated "
           "unit with the untreated unit that looks most like it, so the comparison is between similar units. The effect is the average "
           "outcome gap within pairs. DoWhy states the causal model, runs its own matching estimate and can test it with a placebo "
           "refutation; EconML's doubly robust learner is correct if either the treatment model or the outcome model is right.")
    details = {"att": att, "naive": naive, "ci_low": float(lo), "ci_high": float(hi), "se": se, "n_boot_used": len(boots),
               "n_treated": n_t, "n_matched": int(len(rt)), "caliper_logit": float(cal), "balance": rows, "crosscheck": cross}
    return result("psm", "Propensity score matching", concl, "significant" if sig else "not_significant", key, charts, warnings_, details, how)


# =====================================================================  difference-in-differences
def _did_ols(d: pd.DataFrame, units: np.ndarray | None):
    X = sm.add_constant(d[["treated", "post"]].assign(inter=d["treated"] * d["post"]).to_numpy(float))
    m = sm.OLS(d["y"].to_numpy(float), X)
    if units is not None:
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(units)[0]})
    return m.fit(cov_type="HC1")


def analyze_did(df: pd.DataFrame, outcome: str, group_col: str, treated_value: str, time_col: str, treat_time: float,
                unit_col: str | None = None, alpha: float = 0.05) -> dict:
    require_columns(df, outcome, group_col, time_col, unit_col)
    d = pd.DataFrame({"y": numeric(df[outcome], outcome), "time": numeric(df[time_col], time_col),
                      "treated": (df[group_col].astype(str) == str(treated_value)).astype(int)})
    if unit_col:
        d["unit"] = df[unit_col].astype(str)
    d = d.dropna(subset=["y", "time"])
    if d["treated"].sum() == 0:
        raise UserError(f"No rows have '{treated_value}' in column '{group_col}'. Pick the value that marks the treated group.")
    if (1 - d["treated"]).sum() == 0:
        raise UserError("Every row is in the treated group, so there is no comparison group.")
    d["post"] = (d["time"] >= float(treat_time)).astype(int)
    if d["post"].nunique() < 2:
        raise UserError("The treatment time must fall inside the time range, with data both before and after it.")
    units = d["unit"].to_numpy() if unit_col else None
    fit = _did_ols(d, units)
    est, se, p = float(fit.params[3]), float(fit.bse[3]), float(fit.pvalues[3])
    ci = fit.conf_int(alpha=alpha)[3]
    n_treated_units = d.loc[d["treated"] == 1, "unit"].nunique() if unit_col else None
    n_units = d["unit"].nunique() if unit_col else None

    warnings_: list[str] = []
    # parallel trends: do the groups already trend differently before the change?
    pre = d[d["post"] == 0]
    pt_p = None
    pt_slope = None
    if pre["time"].nunique() >= 3:
        Xp = sm.add_constant(pd.DataFrame({"treated": pre["treated"], "time": pre["time"], "tt": pre["treated"] * pre["time"]}).to_numpy(float))
        mp = sm.OLS(pre["y"].to_numpy(float), Xp)
        fp = (mp.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pre["unit"])[0]}) if unit_col else mp.fit(cov_type="HC1"))
        pt_p, pt_slope = float(fp.pvalues[3]), float(fp.params[3])
        if pt_p < 0.05:
            warnings_.append(f"The groups were already trending differently before the change ({pfmt(pt_p)}). The parallel-trends "
                             "assumption looks doubtful, so the estimate may be biased.")
    else:
        warnings_.append("Fewer than 3 periods before the change, so parallel trends cannot be checked.")

    # placebo-in-space inference (more honest than clustered SEs when few units are treated)
    plc_p, plc = None, []
    if unit_col and n_units and (n_units - n_treated_units) >= 5:
        ctrl = d[d["treated"] == 0]
        for u in ctrl["unit"].unique():
            dd = ctrl.assign(treated=(ctrl["unit"] == u).astype(int))
            f = _did_ols(dd, None)
            plc.append(float(f.params[3]))
        plc = np.asarray(plc)
        plc_p = float((np.sum(np.abs(plc) >= abs(est)) + 1) / (len(plc) + 1))
    if n_treated_units is not None and n_treated_units < 5:
        warnings_.append(f"Only {n_treated_units} treated unit(s): standard errors that cluster by unit are unreliable here. "
                         + ("Rely on the placebo p-value, which compares the effect with what happens to untreated units." if plc_p is not None else
                            "Add a unit column to get placebo-based inference."))
    if unit_col is None:
        warnings_.append("No unit column chosen, so rows are treated as independent. If the same units appear repeatedly, the interval is too narrow.")

    sig = (plc_p < alpha) if plc_p is not None and n_treated_units is not None and n_treated_units < 5 else (p < alpha)
    ymean = d.groupby(["time", "treated"])["y"].mean().unstack()
    tt = sorted(ymean.index)
    prior_t = ymean.loc[ymean.index < float(treat_time), 1].mean()
    concl = (f"Compared with the comparison group, the treated group's outcome changed by {fmt(est)} after the change "
             f"({pct(1 - alpha, 0)} interval {fmt(ci[0])} to {fmt(ci[1])}; {pfmt(p)}"
             + (f"; placebo {pfmt(plc_p)}" if plc_p is not None else "") + "). ")
    if abs(prior_t) > 1e-9:
        concl += f"That is {est / abs(prior_t) * 100:+.1f}% relative to the treated group's pre-change average. "
    if n_treated_units is not None and n_treated_units < 5 and plc_p is not None:
        concl += (f"With so few treated units, the placebo p-value is the one to trust: {pfmt(plc_p)}. ")
    concl += ("The estimate is significant. " if sig else "The data do not clearly distinguish this from no effect. ")
    if pt_p is not None:
        concl += ("Before the change the two groups moved in parallel, which supports the method's key assumption. " if pt_p >= 0.05 else
                  "Before the change the groups were not moving in parallel, so be cautious. ")
    concl += "The result is causal only if, without the change, the treated group would have followed the same trend as the comparison group."
    key = [kn("Difference-in-differences", fmt(est), f"{pct(1 - alpha, 0)} CI {fmt(ci[0])} to {fmt(ci[1])}"),
           kn("p-value (regression)", f"{p:.4f}" if p >= 0.0001 else "< 0.0001", "clustered by unit" if unit_col else "robust SE")]
    if plc_p is not None:
        key.append(kn("p-value (placebo)", f"{plc_p:.3f}", f"vs {len(plc)} untreated units"))
    if pt_p is not None:
        key.append(kn("Pre-trend test", f"{pt_p:.3f}", "high p-value supports parallel trends"))
    if n_treated_units is not None:
        key.append(kn("Units", f"{n_treated_units} treated, {n_units - n_treated_units} comparison"))
    series = [{"name": "Treated group", "points": [{"x": float(x), "y": float(ymean.loc[x, 1])} for x in tt if 1 in ymean.columns and not math.isnan(ymean.loc[x, 1])]},
              {"name": "Comparison group", "points": [{"x": float(x), "y": float(ymean.loc[x, 0])} for x in tt if 0 in ymean.columns and not math.isnan(ymean.loc[x, 0])]}]
    gap = (ymean[1] - ymean[0]).dropna()
    pre_gap = gap[gap.index < float(treat_time)]
    ref_gap = float(pre_gap.iloc[-1]) if len(pre_gap) else 0.0
    charts = [
        {"id": "means", "title": "Average outcome by group over time", "kind": "line", "x_label": time_col, "y_label": outcome,
         "series": series, "vlines": [{"x": float(treat_time), "label": "change"}]},
        {"id": "gap", "title": "Treated minus comparison, relative to the last period before the change", "kind": "line", "x_label": time_col,
         "y_label": "Gap (relative)", "series": [{"name": "Gap", "points": [{"x": float(x), "y": float(v - ref_gap)} for x, v in gap.items()]}],
         "vlines": [{"x": float(treat_time), "label": "change"}], "hlines": [{"y": 0, "label": ""}]},
    ]
    if plc_p is not None:
        h, edges = np.histogram(plc, bins=14)
        charts.append({"id": "placebo", "title": "Placebo test: effects for untreated units (the real estimate is the line)", "kind": "bar",
                       "categories": [f"{(edges[i] + edges[i + 1]) / 2:.3g}" for i in range(len(h))],
                       "series": [{"name": "Untreated units", "values": h.tolist()}], "y_label": "Number of units",
                       "marker": {"x": float(est), "label": "actual"}})
    how = ("Difference-in-differences compares how the outcome changed over time in the treated group with how it changed in a "
           "comparison group. The comparison group's change stands in for what would have happened to the treated group anyway, "
           "so anything left over is attributed to the treatment. It needs the two groups to have been moving in parallel before.")
    details = {"estimate": est, "se": se, "p_value": p, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "pretrend_p": pt_p,
               "pretrend_slope_diff": pt_slope, "placebo_p": plc_p, "n_placebo": int(len(plc)) if plc_p is not None else 0,
               "n_treated_units": n_treated_units, "n_units": n_units}
    return result("did", "Difference-in-differences", concl, "significant" if sig else "not_significant", key, charts, warnings_, details, how)


# =====================================================================  synthetic control
def synth_weights(X0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that make the donors' pre-period path match the treated unit's."""
    J = X0.shape[1]
    w0 = np.full(J, 1.0 / J)
    # work on a common scale so the optimiser tolerance means the same thing for any units
    scale = max(float(np.std(x1)), float(np.mean(np.abs(x1))) * 1e-3, 1e-9)
    X0 = X0 / scale
    x1 = x1 / scale

    def f(w):
        r = x1 - X0 @ w
        return float(r @ r)

    def g(w):
        return -2 * X0.T @ (x1 - X0 @ w)

    res = minimize(f, w0, jac=g, method="SLSQP", bounds=[(0, 1)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(J)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def _rmspe(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def synthetic_control(wide: pd.DataFrame, treated: str, treat_time: float) -> dict:
    """``wide``: rows = time, columns = units. Returns weights, paths, gaps and placebo statistics."""
    times = wide.index.to_numpy(float)
    pre = times < treat_time
    if pre.sum() < 3 or (~pre).sum() < 1:
        raise UserError("Need at least 3 periods before the treatment and 1 after it.")
    donors = [c for c in wide.columns if c != treated]
    if len(donors) < 2:
        raise UserError("Need at least 2 comparison units to build a synthetic control.")
    Y = wide.to_numpy(float)
    ti = list(wide.columns).index(treated)
    D = Y[:, [list(wide.columns).index(c) for c in donors]]
    w = synth_weights(D[pre], Y[pre, ti])
    synth = D @ w
    gap = Y[:, ti] - synth
    pre_r, post_r = _rmspe(gap[pre]), _rmspe(gap[~pre])
    placebos = []
    for j, name in enumerate(donors):
        others = [k for k in range(len(donors)) if k != j]
        if len(others) < 2:
            continue
        wj = synth_weights(D[pre][:, others], D[pre, j])
        gj = D[:, j] - D[:, others] @ wj
        pr, po = _rmspe(gj[pre]), _rmspe(gj[~pre])
        placebos.append({"unit": name, "pre_rmspe": pr, "post_rmspe": po, "ratio": po / pr if pr > 0 else float("inf"), "gap": gj})
    ratio = post_r / pre_r if pre_r > 0 else float("inf")
    p = float((sum(1 for q in placebos if q["ratio"] >= ratio) + 1) / (len(placebos) + 1))
    return {"times": times, "pre": pre, "weights": dict(zip(donors, w)), "treated_path": Y[:, ti], "synthetic": synth, "gap": gap,
            "pre_rmspe": pre_r, "post_rmspe": post_r, "ratio": ratio, "p_value": p, "placebos": placebos,
            "avg_post_gap": float(gap[~pre].mean())}


def analyze_synth(df: pd.DataFrame, outcome: str, unit_col: str, time_col: str, treated_unit: str, treat_time: float) -> dict:
    require_columns(df, outcome, unit_col, time_col)
    d = pd.DataFrame({"y": numeric(df[outcome], outcome), "t": numeric(df[time_col], time_col), "u": df[unit_col].astype(str)}).dropna()
    if str(treated_unit) not in set(d["u"]):
        raise UserError(f"The treated unit '{treated_unit}' is not in column '{unit_col}'.")
    wide = d.pivot_table(index="t", columns="u", values="y", aggfunc="mean")
    warnings_: list[str] = []
    n0 = wide.shape[1]
    wide = wide.dropna(axis=1)
    if str(treated_unit) not in wide.columns:
        raise UserError("The treated unit has gaps in its time series. Synthetic control needs an outcome in every period.")
    if wide.shape[1] < n0:
        warnings_.append(f"{n0 - wide.shape[1]} comparison unit(s) were dropped because they are missing values in some periods.")
    r = synthetic_control(wide, str(treated_unit), float(treat_time))
    times, pre = r["times"], r["pre"]
    top = sorted(r["weights"].items(), key=lambda kv: -kv[1])
    top = [(k, v) for k, v in top if v > 0.005][:8]
    post_synth_mean = float(r["synthetic"][~pre].mean())
    rel = r["avg_post_gap"] / abs(post_synth_mean) * 100 if abs(post_synth_mean) > 1e-9 else float("nan")
    if r["pre_rmspe"] > 0.15 * np.std(r["treated_path"][pre]):
        warnings_.append("The synthetic control does not track the treated unit closely before the change (check the chart), so the "
                         "gap afterwards is hard to attribute to the treatment.")
    if wide.shape[1] < 11:
        warnings_.append("Few comparison units: the placebo p-value cannot be very small (smallest possible is 1 / number of units).")
    sig = r["p_value"] <= 0.1
    concl = (f"After {treat_time:g}, {treated_unit}'s outcome averaged {fmt(abs(r['avg_post_gap']))} "
             f"{'above' if r['avg_post_gap'] >= 0 else 'below'} what a weighted blend of other units predicts"
             + (f" ({rel:+.1f}%)" if math.isfinite(rel) else "") + ". The blend is mostly "
             + ", ".join(f"{k} ({v * 100:.0f}%)" for k, v in top[:3]) + ". ")
    concl += (f"To judge whether that gap is unusual, the same procedure was applied to each of the {len(r['placebos'])} other units as if "
              f"it were the treated one. {treated_unit}'s post-to-pre error ratio is {r['ratio']:.1f}, and {r['p_value'] * 100:.0f}% of units "
              f"({pfmt(r['p_value'])}) show a ratio at least that large. ")
    concl += ("That makes the gap stand out from ordinary fluctuations. " if sig else "That means gaps this large also appear often by chance, so the effect is not clearly distinguishable from noise. ")
    concl += "The estimate is credible only if the blend fits the treated unit well before the change and nothing else changed for it at the same time."
    key = [kn("Average gap after the change", fmt(r["avg_post_gap"]), "treated minus synthetic"),
           kn("Placebo p-value", f"{r['p_value']:.3f}", f"{len(r['placebos'])} placebo units"),
           kn("Fit before the change (RMSPE)", fmt(r["pre_rmspe"]), "lower = better match"),
           kn("Post / pre error ratio", f"{r['ratio']:.1f}", "large = effect stands out")]
    charts = [
        {"id": "paths", "title": f"{treated_unit} versus its synthetic control", "kind": "line", "x_label": time_col, "y_label": outcome,
         "series": [{"name": str(treated_unit), "points": [{"x": float(x), "y": float(v)} for x, v in zip(times, r["treated_path"])]},
                    {"name": "Synthetic control", "dashed": True, "points": [{"x": float(x), "y": float(v)} for x, v in zip(times, r["synthetic"])]}],
         "vlines": [{"x": float(treat_time), "label": "change"}]},
    ]
    keep_pl = [q for q in r["placebos"] if q["pre_rmspe"] <= 5 * max(r["pre_rmspe"], 1e-9)][:40]
    gap_series = [{"name": "Placebo units" if i == 0 else "", "muted": True, "points": [{"x": float(x), "y": float(v)} for x, v in zip(times, q["gap"])]}
                  for i, q in enumerate(keep_pl)]
    gap_series.append({"name": f"{treated_unit} (actual gap)", "strong": True, "points": [{"x": float(x), "y": float(v)} for x, v in zip(times, r["gap"])]})
    charts.append({"id": "gap", "title": "Gap: actual minus synthetic, with placebo gaps in grey", "kind": "line", "x_label": time_col, "y_label": "Gap",
                   "series": gap_series, "vlines": [{"x": float(treat_time), "label": "change"}], "hlines": [{"y": 0, "label": ""}]})
    charts.append({"id": "weights", "title": "Which units form the synthetic control", "kind": "bar", "categories": [k for k, _ in top],
                   "series": [{"name": "Weight", "values": [v for _, v in top]}], "y_label": "Weight"})
    how = ("Synthetic control builds a stand-in for the treated unit out of a weighted mix of untreated units, chosen so the mix "
           "follows the treated unit's history as closely as possible before the change. After the change, the stand-in shows what "
           "would probably have happened without it. Re-running the same procedure on every untreated unit shows how large a gap "
           "can appear by chance, which gives a p-value even with a single treated unit.")
    details = {"avg_post_gap": r["avg_post_gap"], "relative_gap_pct": rel, "p_value": r["p_value"], "pre_rmspe": r["pre_rmspe"],
               "post_rmspe": r["post_rmspe"], "ratio": r["ratio"], "weights": {k: float(v) for k, v in r["weights"].items() if v > 1e-4},
               "n_placebos": len(r["placebos"])}
    return result("synth", "Synthetic control", concl, "significant" if sig else "not_significant", key, charts, warnings_, details, how)
