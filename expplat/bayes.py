"""Bayesian A/B analysis with PyMC: the probability that B beats A.

Binary metric  -> Beta-Binomial model (Beta(1, 1) priors unless you choose others).
Numeric metric -> Normal model with unknown mean and standard deviation per group,
                  fitted on the sufficient statistics (sample mean and sum of squares).
                  That is the exact same posterior as using every row, and it keeps the
                  run time the same for a thousand rows or a million.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pymc as pm

from .ab_tests import _binary_outcome, srm_check, _srm_warning
from .common import UserError, fmt, kn, numeric, pct, require_columns, result, split_groups


# ---------------------------------------------------------------- diagnostics
def _split_rhat(draws: np.ndarray) -> float:
    """Gelman-Rubin R-hat on split chains; draws has shape (chains, draws)."""
    c, n = draws.shape
    h = n // 2
    chains = np.concatenate([draws[:, :h], draws[:, h:2 * h]], axis=0)
    m, n2 = chains.shape
    W = chains.var(axis=1, ddof=1).mean()
    B = n2 * chains.mean(axis=1).var(ddof=1)
    if W <= 0:
        return float("nan")
    var_hat = (n2 - 1) / n2 * W + B / n2
    return float(math.sqrt(var_hat / W))


def _density(samples: np.ndarray, lo: float, hi: float, bins: int = 60) -> list[dict]:
    edges = np.linspace(lo, hi, bins + 1)
    h, _ = np.histogram(samples, bins=edges, density=True)
    mid = (edges[:-1] + edges[1:]) / 2
    return [{"x": float(m), "y": float(v)} for m, v in zip(mid, h)]


def _summ(diff: np.ndarray, rel: np.ndarray | None, pa: np.ndarray, pb: np.ndarray, min_effect: float, rhat: float,
          ess: float | None) -> dict:
    out = {
        "prob_b_beats_a": float((diff > 0).mean()),
        "prob_b_beats_a_by_min": float((diff > min_effect).mean()) if min_effect else None,
        "diff_mean": float(diff.mean()), "diff_low": float(np.percentile(diff, 2.5)), "diff_high": float(np.percentile(diff, 97.5)),
        "expected_loss_choose_b": float(np.maximum(-diff, 0).mean()),
        "expected_loss_choose_a": float(np.maximum(diff, 0).mean()),
        "mean_a": float(pa.mean()), "mean_b": float(pb.mean()), "rhat_max": rhat, "ess_min": ess,
    }
    if rel is not None:
        out.update(rel_median=float(np.median(rel)), rel_low=float(np.percentile(rel, 2.5)), rel_high=float(np.percentile(rel, 97.5)))
    return out


def _sample(model: pm.Model, draws: int, tune: int, chains: int, seed: int):
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed, progressbar=False,
                         compute_convergence_checks=False)


def _diagnostics(idata, names: list[str]) -> tuple[float, float | None]:
    rh, ess = [], []
    for n in names:
        v = idata.posterior[n].values
        rh.append(_split_rhat(v))
    try:  # effective sample size via arviz when available
        import arviz as az
        for n in names:
            ess.append(float(az.ess(idata)[n].values))
    except Exception:  # pragma: no cover - diagnostics are best effort
        ess = []
    return float(np.nanmax(rh)), (float(min(ess)) if ess else None)


# ---------------------------------------------------------------- binary
def bayes_binary(sa: int, na: int, sb: int, nb: int, prior_a: float = 1.0, prior_b: float = 1.0, min_effect: float = 0.0,
                 draws: int = 2000, tune: int = 1000, chains: int = 2, seed: int = 42) -> dict:
    with pm.Model() as model:
        pa = pm.Beta("pa", prior_a, prior_b)
        pb = pm.Beta("pb", prior_a, prior_b)
        pm.Binomial("ya", n=na, p=pa, observed=sa)
        pm.Binomial("yb", n=nb, p=pb, observed=sb)
    idata = _sample(model, draws, tune, chains, seed)
    a = idata.posterior["pa"].values.reshape(-1)
    b = idata.posterior["pb"].values.reshape(-1)
    rhat, ess = _diagnostics(idata, ["pa", "pb"])
    res = _summ(b - a, b / a - 1, a, b, min_effect, rhat, ess)
    res["_a"], res["_b"] = a, b
    return res


# ---------------------------------------------------------------- numeric
def bayes_numeric(a: np.ndarray, b: np.ndarray, min_effect: float = 0.0, draws: int = 2000, tune: int = 1000,
                  chains: int = 2, seed: int = 42, full_data: bool = False) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pooled = np.concatenate([a, b])
    centre, scale = float(pooled.mean()), float(pooled.std(ddof=1))
    if scale <= 0:
        raise UserError("All values are identical, so there is nothing to compare.")
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", centre, 5 * scale)
        mu_b = pm.Normal("mu_b", centre, 5 * scale)
        sd_a = pm.HalfNormal("sd_a", 2 * scale)
        sd_b = pm.HalfNormal("sd_b", 2 * scale)
        if full_data:
            pm.Normal("ya", mu_a, sd_a, observed=a)
            pm.Normal("yb", mu_b, sd_b, observed=b)
        else:
            for tag, x, mu, sd in (("a", a, mu_a, sd_a), ("b", b, mu_b, sd_b)):
                n = len(x)
                ss = float(((x - x.mean()) ** 2).sum())
                pm.Normal(f"ybar_{tag}", mu, sd / math.sqrt(n), observed=float(x.mean()))
                pm.Gamma(f"ss_{tag}", alpha=(n - 1) / 2, beta=1 / (2 * sd ** 2), observed=ss)
    idata = _sample(model, draws, tune, chains, seed)
    ma = idata.posterior["mu_a"].values.reshape(-1)
    mb = idata.posterior["mu_b"].values.reshape(-1)
    rhat, ess = _diagnostics(idata, ["mu_a", "mu_b"])
    rel = (mb / ma - 1) if (ma > 0).all() else None
    res = _summ(mb - ma, rel, ma, mb, min_effect, rhat, ess)
    res["_a"], res["_b"] = ma, mb
    return res


# ---------------------------------------------------------------- API entry point
def analyze_bayes(df: pd.DataFrame, group_col: str, metric_col: str, control: str, treatment: str,
                  metric_type: str = "auto", success_value: str | None = None, min_effect: float = 0.0,
                  transform: str = "none", draws: int = 2000, seed: int = 42, expected_ratio: float = 1.0) -> dict:
    require_columns(df, group_col, metric_col)
    ga, gb = split_groups(df, group_col, control, treatment)
    both = pd.concat([ga, gb])
    binary = None
    if metric_type in ("auto", "binary"):
        binary = _binary_outcome(both[metric_col], success_value)
        if metric_type == "binary" and binary is None:
            raise UserError("The metric has more than two different values, so it cannot be treated as yes/no.")
    warnings: list[str] = []
    if binary is not None:
        oa, ob = binary.loc[ga.index].dropna(), binary.loc[gb.index].dropna()
        srm = srm_check(len(oa), len(ob), expected_ratio)
        warnings += _srm_warning(srm)
        r = bayes_binary(int(oa.sum()), len(oa), int(ob.sum()), len(ob), min_effect=min_effect, draws=draws, seed=seed)
        kind, unit_mult, unit = "binary", 100.0, "percentage points"
        label_a, label_b = f"{pct(oa.mean(), 2)}", f"{pct(ob.mean(), 2)}"
        n_a, n_b = len(oa), len(ob)
    else:
        xa = numeric(ga[metric_col], metric_col).dropna().to_numpy()
        xb = numeric(gb[metric_col], metric_col).dropna().to_numpy()
        srm = srm_check(len(xa), len(xb), expected_ratio)
        warnings += _srm_warning(srm)
        if transform == "log1p":
            if (xa < 0).any() or (xb < 0).any():
                raise UserError("The log transform needs values that are zero or above.")
            xa, xb = np.log1p(xa), np.log1p(xb)
        elif transform != "none":
            raise UserError("Transform must be 'none' or 'log1p'.")
        r = bayes_numeric(xa, xb, min_effect=min_effect, draws=draws, seed=seed)
        kind, unit_mult, unit = "numeric", 1.0, "in the metric" if transform == "none" else "in log(1 + metric)"
        label_a, label_b = fmt(xa.mean()), fmt(xb.mean())
        n_a, n_b = len(xa), len(xb)
        if transform == "none" and (abs(pd.Series(xa).skew()) > 3 or abs(pd.Series(xb).skew()) > 3):
            warnings.append("The metric is very skewed. The Normal model describes the average poorly here: try the log transform.")
    if r["rhat_max"] and r["rhat_max"] > 1.01:
        warnings.append(f"The sampler did not fully converge (R-hat {r['rhat_max']:.3f}). Treat the numbers as approximate.")
    if min(n_a, n_b) < 30:
        warnings.append("Very few users: the result depends strongly on the prior, so read it with caution.")

    p = r["prob_b_beats_a"]
    d = unit_mult
    concl = (f"There is a {pct(p, 1)} probability that the treatment (B) is better than control (A). "
             f"The most likely size of the difference is {r['diff_mean'] * d:+.2f} {unit} "
             f"(95% credible interval {r['diff_low'] * d:+.2f} to {r['diff_high'] * d:+.2f}). ")
    if min_effect:
        concl += (f"The probability that B beats A by more than your minimum worthwhile effect "
                  f"({min_effect * d:g} {unit}) is {pct(r['prob_b_beats_a_by_min'], 1)}. ")
    def loss(v: float) -> str:
        v = v * d
        return "less than 0.001" if v < 0.001 else f"{v:.3f}"

    concl += (f"If you ship B and it is actually worse, you would lose {loss(r['expected_loss_choose_b'])} {unit} on average; "
              f"if you stay with A and B is actually better, you would give up {loss(r['expected_loss_choose_a'])} {unit}. ")
    if p >= 0.95:
        verdict = "significant"
        concl += "The evidence favours B strongly."
    elif p <= 0.05:
        verdict = "significant"
        concl += "The evidence favours A strongly (B looks worse)."
    else:
        verdict = "inconclusive"
        concl += "The evidence is not decisive either way yet. A common rule of thumb is to act at 95% or above."
    concl += " This is a probability about the effect given your data and prior, which is a different statement from a p-value."

    lo = min(np.percentile(r["_a"], 0.1), np.percentile(r["_b"], 0.1))
    hi = max(np.percentile(r["_a"], 99.9), np.percentile(r["_b"], 99.9))
    diff = r["_b"] - r["_a"]
    dlo, dhi = np.percentile(diff, [0.1, 99.9])
    xmult = unit_mult if kind == "binary" else 1.0
    charts = [
        {"id": "post", "title": "What we believe each group's true value is", "kind": "line", "x_label": "% rate" if kind == "binary" else "mean",
         "y_label": "Density", "series": [
             {"name": f"A (control), observed {label_a}", "fill": True, "points": [{"x": q["x"] * xmult, "y": q["y"]} for q in _density(r["_a"], lo, hi)]},
             {"name": f"B (treatment), observed {label_b}", "fill": True, "points": [{"x": q["x"] * xmult, "y": q["y"]} for q in _density(r["_b"], lo, hi)]}]},
        {"id": "diff", "title": "Difference B minus A", "kind": "line", "x_label": f"Difference ({unit})", "y_label": "Density",
         "series": [{"name": "Posterior of the difference", "fill": True,
                     "points": [{"x": q["x"] * d, "y": q["y"]} for q in _density(diff, dlo, dhi)]}],
         "vlines": [{"x": 0.0, "label": "no difference"}]},
    ]
    key = [kn("Probability B beats A", pct(p, 1)), kn("Expected difference", f"{r['diff_mean'] * d:+.3f}", unit),
           kn("95% credible interval", f"{r['diff_low'] * d:+.3f} to {r['diff_high'] * d:+.3f}", unit),
           kn("Expected loss if you ship B", ("< 0.001" if r['expected_loss_choose_b'] * d < 0.001 else f"{r['expected_loss_choose_b'] * d:.3f}"), unit),
           kn("Sampler check (R-hat)", f"{r['rhat_max']:.3f}", "should be below 1.01")]
    if "rel_median" in r:
        key.append(kn("Relative change", f"{r['rel_median'] * 100:+.1f}%", f"95% interval {r['rel_low'] * 100:+.1f}% to {r['rel_high'] * 100:+.1f}%"))
    how = ("A Bayesian analysis starts with a weak prior belief about each group, updates it with the data (PyMC draws thousands of "
           "plausible values for each group's true rate or mean), and then simply counts how often B's value is above A's. "
           "Expected loss turns that into a decision: how much you would lose, on average, if you picked the wrong one.")
    details = {k: v for k, v in r.items() if not k.startswith("_")}
    details.update(srm=srm, kind=kind, draws=draws * 2, transform=transform)
    return result("bayes", "Bayesian A/B analysis", concl, verdict, key, charts, warnings, details, how)
