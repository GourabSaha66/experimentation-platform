"""Sequential testing you can peek at: the mixture sequential probability ratio test (mSPRT).

A normal p-value is only valid if you look once, at a sample size fixed in advance.
Looking every day and stopping when p < 0.05 pushes the false-alarm rate far above 5%.
The mSPRT gives an "always-valid" p-value and a confidence sequence that stay correct
however often you look, and whenever you decide to stop (Johari et al., 2017; Howard
et al., 2021 for the confidence-sequence form).

For the difference in means theta with estimated standard error sqrt(V) and a
Normal(0, tau^2) mixing distribution over theta, the likelihood ratio is

    Lambda = sqrt(V / (V + tau^2)) * exp( tau^2 * theta_hat^2 / (2 V (V + tau^2)) )

Reject at level alpha as soon as Lambda >= 1/alpha. The always-valid p-value is
min(1, 1 / max over looks of Lambda). Variances are estimated from the data, so the
guarantee is asymptotic: very small samples are not covered.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .common import UserError, fmt, kn, numeric, pct, pfmt, require_columns, result


def msprt_path(x: np.ndarray, is_treat: np.ndarray, tau: float, alpha: float = 0.05, n_looks: int = 60,
               min_per_group: int = 20) -> pd.DataFrame:
    """Evaluate the test after each of ``n_looks`` evenly spaced points of the data stream.

    ``x`` holds the metric in arrival order; ``is_treat`` is 1 for treatment rows, 0 for control.
    """
    x = np.asarray(x, float)
    t = np.asarray(is_treat, bool)
    n = len(x)
    c_n = np.cumsum(~t)
    t_n = np.cumsum(t)
    c_s = np.cumsum(np.where(~t, x, 0.0))
    t_s = np.cumsum(np.where(t, x, 0.0))
    c_ss = np.cumsum(np.where(~t, x * x, 0.0))
    t_ss = np.cumsum(np.where(t, x * x, 0.0))

    ok = np.where((c_n >= min_per_group) & (t_n >= min_per_group))[0]
    if len(ok) == 0:
        raise UserError(f"Need at least {min_per_group} users in each group before the sequential test can start.")
    idx = np.unique(np.linspace(ok[0], n - 1, n_looks).astype(int))

    na, nb = c_n[idx].astype(float), t_n[idx].astype(float)
    ma, mb = c_s[idx] / na, t_s[idx] / nb
    va = np.maximum((c_ss[idx] - na * ma ** 2) / (na - 1), 1e-12)
    vb = np.maximum((t_ss[idx] - nb * mb ** 2) / (nb - 1), 1e-12)
    V = va / na + vb / nb
    theta = mb - ma
    tau2 = tau ** 2
    log_lam = 0.5 * np.log(V / (V + tau2)) + tau2 * theta ** 2 / (2 * V * (V + tau2))
    p_raw = np.minimum(1.0, np.exp(-log_lam))
    p_av = np.minimum.accumulate(p_raw)
    half = np.sqrt(V * (V + tau2) / tau2 * np.log((V + tau2) / (V * alpha ** 2)))
    lo = np.maximum.accumulate(theta - half)
    hi = np.minimum.accumulate(theta + half)
    z = theta / np.sqrt(V)
    p_naive = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return pd.DataFrame({"n_total": idx + 1, "n_control": na.astype(int), "n_treatment": nb.astype(int),
                         "mean_control": ma, "mean_treatment": mb, "estimate": theta, "se": np.sqrt(V),
                         "p_always_valid": p_av, "p_naive": p_naive,
                         "cs_low": lo, "cs_high": hi, "ci_low_now": theta - half, "ci_high_now": theta + half})


def default_tau(x: np.ndarray) -> float:
    """Mixing sd for the effect: 10% of the metric's standard deviation unless the user says otherwise."""
    return 0.1 * float(np.std(x, ddof=1))


def analyze_sequential(df: pd.DataFrame, group_col: str, metric_col: str, control: str, treatment: str,
                       alpha: float = 0.05, order_col: str | None = None, expected_effect: float | None = None,
                       n_looks: int = 60) -> dict:
    require_columns(df, group_col, metric_col, order_col)
    g = df[group_col].astype(str)
    d = df[g.isin([str(control), str(treatment)])].copy()
    if d.empty or d[group_col].astype(str).nunique() < 2:
        raise UserError(f"Could not find both groups '{control}' and '{treatment}' in column '{group_col}'.")
    d["_x"] = numeric(d[metric_col], metric_col)
    d = d.dropna(subset=["_x"])
    warnings: list[str] = []
    if order_col:
        try:
            d = d.assign(_o=pd.to_datetime(d[order_col]))
        except (ValueError, TypeError):
            d = d.assign(_o=pd.to_numeric(d[order_col], errors="coerce"))
        d = d.sort_values("_o", kind="stable")
    else:
        warnings.append("No time column chosen, so the rows are assumed to be in the order users arrived.")
    x = d["_x"].to_numpy()
    is_t = (d[group_col].astype(str) == str(treatment)).to_numpy()
    tau = float(expected_effect) if expected_effect else default_tau(x)
    if tau <= 0:
        raise UserError("The expected effect size must be above 0 (or leave it blank for the default).")
    path = msprt_path(x, is_t, tau, alpha, n_looks)
    last = path.iloc[-1]
    stop_rows = path[path["p_always_valid"] <= alpha]
    stopped = len(stop_rows) > 0
    first = stop_rows.iloc[0] if stopped else None
    naive_hit = path[path["p_naive"] <= alpha]
    sig_now = last["p_always_valid"] <= alpha

    if stopped:
        concl = (f"You can safely stop: the evidence crossed the significance threshold after {int(first['n_total']):,} users "
                 f"(always-valid {pfmt(first['p_always_valid'])}). Using all {int(last['n_total']):,} users so far, treatment minus "
                 f"control is {fmt(last['estimate'])}, and the {pct(1 - alpha, 0)} confidence sequence runs from {fmt(last['cs_low'])} to "
                 f"{fmt(last['cs_high'])}. Because this method is built for repeated looks, stopping when it first crosses is valid. "
                 f"The estimate itself tends to overstate the effect a little when you stop early, so treat the size with care.")
        verdict = "significant"
    else:
        concl = (f"No decision yet. After {int(last['n_total']):,} users the difference is {fmt(last['estimate'])} and the "
                 f"{pct(1 - alpha, 0)} confidence sequence still includes zero ({fmt(last['cs_low'])} to {fmt(last['cs_high'])}); the "
                 f"always-valid p-value is {pfmt(last['p_always_valid'])}. You can keep collecting data and look again without "
                 f"inflating your false-alarm rate. If you stop now, say 'no significant effect found yet', not 'no effect'.")
        verdict = "inconclusive"
    if len(naive_hit) and not stopped:
        warnings.append("A normal t-test would have shown p < 0.05 at some look, but the sequential test did not. This is "
                        "exactly the false alarm that peeking at ordinary p-values produces.")
    key = [kn("Users so far", f"{int(last['n_total']):,}", f"{int(last['n_control']):,} control, {int(last['n_treatment']):,} treatment"),
           kn("Difference", fmt(last["estimate"]), "treatment minus control"),
           kn("Always-valid p-value", f"{last['p_always_valid']:.4f}" if last["p_always_valid"] >= 0.0001 else "< 0.0001", f"stop if below {alpha}"),
           kn("Confidence sequence", f"{fmt(last['cs_low'])} to {fmt(last['cs_high'])}", f"{pct(1 - alpha, 0)}, valid at every look"),
           kn("Stopped at", f"{int(first['n_total']):,} users" if stopped else "not yet", "first look past the threshold")]

    pts_av = [{"x": int(r.n_total), "y": float(max(r.p_always_valid, 1e-6))} for r in path.itertuples()]
    pts_nv = [{"x": int(r.n_total), "y": float(max(r.p_naive, 1e-6))} for r in path.itertuples()]
    charts = [
        {"id": "pvals", "title": "p-values at each look", "kind": "line", "x_label": "Users so far", "y_label": "p-value (log scale)",
         "log_y": True, "series": [{"name": "Always-valid (safe to peek)", "points": pts_av},
                                   {"name": "Ordinary p-value (not safe when peeking)", "points": pts_nv, "dashed": True}],
         "hlines": [{"y": alpha, "label": f"alpha = {alpha}"}]},
        {"id": "cs", "title": "Estimated difference with confidence sequence", "kind": "line", "x_label": "Users so far",
         "y_label": "Treatment minus control",
         "series": [{"name": "Estimate", "points": [{"x": int(r.n_total), "y": float(r.estimate), "lo": float(r.cs_low), "hi": float(r.cs_high)} for r in path.itertuples()]}],
         "hlines": [{"y": 0, "label": "no effect"}]},
    ]
    how = ("The mixture sequential probability ratio test keeps a running likelihood ratio between 'some effect' and 'no effect'. "
           "Because the guarantee is on the whole sequence, you may look as often as you like and stop the moment the "
           "evidence passes the bar, with the false-alarm rate still held at alpha. The price is slightly less power than a "
           "single look at a fixed sample size.")
    details = {"tau": tau, "alpha": alpha, "looks": path.round(8).to_dict(orient="records"),
               "stopped": bool(stopped), "stopped_at": int(first["n_total"]) if stopped else None}
    return result("sequential", "Sequential test (safe to peek)", concl, verdict, key, charts, warnings, details, how)
