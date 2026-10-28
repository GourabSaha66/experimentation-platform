"""Power and sample-size calculator for two-group experiments.

Two kinds of metric are supported:

* ``means``       - a continuous metric (revenue per user, minutes played). Needs the
                    metric's standard deviation.
* ``proportions`` - a conversion-style metric (clicked / did not click). Needs the
                    baseline rate.

For either kind you can solve for the sample size, the power, or the smallest effect
you could reliably detect (the "minimum detectable effect", MDE).
"""
from __future__ import annotations

import math

import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from .common import UserError, fmt, kn, pct, result


def _check_common(alpha: float, power: float | None, ratio: float) -> None:
    if not 0 < alpha < 1:
        raise UserError("Significance level (alpha) must be between 0 and 1, for example 0.05.")
    if power is not None and not 0 < power < 1:
        raise UserError("Power must be between 0 and 1, for example 0.8.")
    if ratio <= 0:
        raise UserError("The group size ratio must be above 0 (1 means equal groups).")


def _p2_from_h(p1: float, h: float) -> float:
    """Invert Cohen's h: the treatment rate that is h away from p1."""
    return math.sin((h + 2 * math.asin(math.sqrt(p1))) / 2) ** 2


def plan(kind: str, solve_for: str, *, alpha: float = 0.05, power: float | None = 0.8,
         ratio: float = 1.0, alternative: str = "two-sided",
         baseline_rate: float | None = None, sd: float | None = None,
         mde: float | None = None, mde_type: str = "relative",
         n_per_group: int | None = None, daily_users: int | None = None) -> dict:
    """Solve for one of sample size / power / mde and return numbers, text and charts."""
    _check_common(alpha, power if solve_for != "power" else None, ratio)
    if kind not in ("means", "proportions"):
        raise UserError("Metric type must be 'means' or 'proportions'.")
    if solve_for not in ("sample_size", "power", "mde"):
        raise UserError("Solve for 'sample_size', 'power' or 'mde'.")
    if alternative not in ("two-sided", "larger"):
        raise UserError("Alternative must be 'two-sided' or 'larger'.")
    if solve_for in ("power", "mde") and not n_per_group:
        raise UserError("Enter how many users you have per group (control group size).")
    if solve_for in ("sample_size", "power") and mde is None:
        raise UserError("Enter the smallest effect you care about detecting.")

    if kind == "means":
        if not sd or sd <= 0:
            raise UserError("Enter the metric's standard deviation (it must be above 0).")
        engine = TTestIndPower()
        p1 = None

        def to_d(eff_abs: float) -> float:
            return eff_abs / sd

        def from_d(d: float) -> float:
            return d * sd

        def abs_effect(m: float) -> float:
            return m
    else:
        if baseline_rate is None or not 0 < baseline_rate < 1:
            raise UserError("Baseline rate must be between 0 and 1, e.g. 0.10 for a 10% conversion rate.")
        engine = NormalIndPower()
        p1 = baseline_rate

        def abs_effect(m: float) -> float:
            return m * p1 if mde_type == "relative" else m

        def to_d(eff_abs: float) -> float:
            p2 = p1 + eff_abs
            if not 0 < p2 < 1:
                raise UserError("That effect would push the rate outside 0%-100%. Pick a smaller effect.")
            return abs(float(proportion_effectsize(p2, p1)))

        def from_d(d: float) -> float:
            p2 = _p2_from_h(p1, d)
            return p2 - p1

    alt = "two-sided" if alternative == "two-sided" else "larger"

    def solve_n(eff_abs: float, pw: float) -> float:
        d = to_d(abs(eff_abs))
        return float(engine.solve_power(effect_size=d, nobs1=None, alpha=alpha, power=pw, ratio=ratio, alternative=alt))

    def power_at(eff_abs: float, n1: float) -> float:
        d = to_d(abs(eff_abs))
        return float(engine.power(effect_size=d, nobs1=n1, alpha=alpha, ratio=ratio, alternative=alt))

    out: dict = {"kind": kind, "solve_for": solve_for, "alpha": alpha, "ratio": ratio, "alternative": alternative}
    warnings: list[str] = []
    charts: list[dict] = []

    if solve_for == "sample_size":
        eff = abs_effect(mde)
        n1 = math.ceil(solve_n(eff, power))
        n2 = math.ceil(n1 * ratio)
        out.update(n_control=n1, n_treatment=n2, n_total=n1 + n2, power=power, effect_absolute=eff)
        # chart: required n for a range of effect sizes
        grid = np.linspace(0.5 * abs(eff), 2.0 * abs(eff), 14)
        pts = [{"x": float(g), "y": float(math.ceil(solve_n(g, power)) * (1 + ratio))} for g in grid]
        charts.append({"id": "n_vs_effect", "title": "Total users needed for different effect sizes", "kind": "line",
                       "x_label": "Effect to detect (absolute)", "y_label": "Total users (both groups)",
                       "series": [{"name": f"{pct(power, 0)} power", "points": pts}],
                       "vlines": [{"x": float(abs(eff)), "label": "your target"}]})
    elif solve_for == "power":
        eff = abs_effect(mde)
        pw = power_at(eff, n_per_group)
        n2 = math.ceil(n_per_group * ratio)
        out.update(n_control=int(n_per_group), n_treatment=n2, n_total=int(n_per_group) + n2, power=pw, effect_absolute=eff)
        ns = np.linspace(max(5, 0.2 * n_per_group), 2.0 * n_per_group, 16)
        pts = [{"x": float(n), "y": power_at(eff, n)} for n in ns]
        charts.append({"id": "power_vs_n", "title": "Power as sample size grows", "kind": "line",
                       "x_label": "Users per group (control)", "y_label": "Power",
                       "series": [{"name": "power", "points": pts}],
                       "vlines": [{"x": float(n_per_group), "label": "your n"}],
                       "hlines": [{"y": 0.8, "label": "80%"}]})
    else:
        d = float(engine.solve_power(effect_size=None, nobs1=n_per_group, alpha=alpha, power=power, ratio=ratio, alternative=alt))
        eff = from_d(d)
        n2 = math.ceil(n_per_group * ratio)
        out.update(n_control=int(n_per_group), n_treatment=n2, n_total=int(n_per_group) + n2, power=power, effect_absolute=eff)
        ns = np.linspace(max(5, 0.2 * n_per_group), 3.0 * n_per_group, 16)
        pts = []
        for n in ns:
            dd = float(engine.solve_power(effect_size=None, nobs1=n, alpha=alpha, power=power, ratio=ratio, alternative=alt))
            pts.append({"x": float(n), "y": float(from_d(dd))})
        charts.append({"id": "mde_vs_n", "title": "Smallest detectable effect as sample size grows", "kind": "line",
                       "x_label": "Users per group (control)", "y_label": "Minimum detectable effect (absolute)",
                       "series": [{"name": "MDE", "points": pts}],
                       "vlines": [{"x": float(n_per_group), "label": "your n"}]})

    eff = out["effect_absolute"]
    if kind == "proportions":
        out["baseline_rate"] = p1
        out["treatment_rate"] = p1 + eff
        out["relative_effect"] = eff / p1
    else:
        out["relative_effect"] = None
    if daily_users:
        out["days_needed"] = math.ceil(out["n_total"] / daily_users)

    # text
    if kind == "proportions":
        eff_txt = (f"an absolute change of {abs(eff) * 100:.2f} percentage points "
                   f"({pct(p1, 1)} to {pct(p1 + eff, 1)}, a {abs(eff) / p1 * 100:.1f}% relative change)")
    else:
        eff_txt = f"a difference of {fmt(abs(eff))} in the average (standard deviation {fmt(sd)})"
    sided = "two-sided" if alternative == "two-sided" else "one-sided"
    if solve_for == "sample_size":
        concl = (f"To detect {eff_txt} with {pct(power, 0)} power at a {pct(alpha, 0)} significance level ({sided}), "
                 f"you need about {out['n_control']:,} users in the control group and {out['n_treatment']:,} in the "
                 f"treatment group, {out['n_total']:,} in total.")
    elif solve_for == "power":
        concl = (f"With {out['n_control']:,} control and {out['n_treatment']:,} treatment users, you have a "
                 f"{pct(out['power'], 0)} chance of detecting {eff_txt} at a {pct(alpha, 0)} significance level ({sided}).")
        if out["power"] < 0.8:
            warnings.append("Power is below the usual 80% target: a real effect of this size would often be missed.")
    else:
        concl = (f"With {out['n_control']:,} control and {out['n_treatment']:,} treatment users, you can reliably "
                 f"({pct(power, 0)} of the time) detect {eff_txt}. Smaller effects will often go unnoticed.")
    if daily_users:
        concl += f" At {daily_users:,} users a day that is about {out['days_needed']} days of traffic."
    if solve_for == "sample_size":
        warnings.append("If you plan to look at results before the planned end, use the sequential test in the Analyse "
                        "tab. Checking a normal test repeatedly makes false alarms much more likely.")

    key = [kn("Control users", f"{out['n_control']:,}"), kn("Treatment users", f"{out['n_treatment']:,}"),
           kn("Power", pct(out["power"], 0) if solve_for != "power" else pct(out["power"], 1)),
           kn("Effect", f"{abs(eff) * 100:.2f} pp" if kind == "proportions" else fmt(abs(eff)),
              "absolute" if kind == "proportions" else "")]
    if daily_users:
        key.append(kn("Days of traffic", f"{out['days_needed']}"))
    how = ("The calculator uses the standard normal-approximation formulas (statsmodels). It finds the sample size at which "
           "a real effect of your chosen size would be detected with the stated probability (power) while keeping false "
           "alarms at the stated rate (alpha). For conversion metrics the effect is converted to Cohen's h first.")
    return result("power", "Sample size and power", concl, "info", key, charts, warnings, out, how)
