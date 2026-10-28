"""Small helpers shared by every analysis module."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class UserError(ValueError):
    """Raised when the uploaded data or the chosen settings cannot be used.

    The message is written for the person using the tool, so the API passes it
    straight back instead of a stack trace.
    """


def clean(obj: Any) -> Any:
    """Make a nested structure JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def require_columns(df: pd.DataFrame, *cols: str | None) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise UserError(
            "These columns are not in your file: " + ", ".join(f"'{c}'" for c in missing)
            + ". Columns found: " + ", ".join(f"'{c}'" for c in df.columns[:30]) + "."
        )


def split_groups(df: pd.DataFrame, group_col: str, control: str, treatment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (control rows, treatment rows), comparing labels as text."""
    require_columns(df, group_col)
    g = df[group_col].astype(str)
    a = df[g == str(control)]
    b = df[g == str(treatment)]
    if len(a) == 0 or len(b) == 0:
        found = ", ".join(sorted(g.unique())[:10])
        raise UserError(
            f"Could not find both groups in column '{group_col}'. Looking for '{control}' and "
            f"'{treatment}'. Values found: {found}."
        )
    return a, b


def numeric(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().sum() == 0:
        raise UserError(f"Column '{name}' has no numbers in it, so it cannot be analysed as a number.")
    return out


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def fmt(x: float, digits: int = 3) -> str:
    """Readable number: thousands separators for big values, sig figs for small."""
    if x is None or not math.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.{max(digits - 1, 1)}f}"
    return f"{x:.{digits}g}"


def pfmt(p: float) -> str:
    if p is None or not math.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def result(method: str, title: str, conclusion: str, verdict: str, key_numbers: list, charts: list,
           warnings: list, details: dict, how: str) -> dict:
    """Every analysis returns this same shape, so the front end can render any of them."""
    return clean({
        "method": method,
        "title": title,
        "conclusion": conclusion,
        "verdict": verdict,  # significant | not_significant | inconclusive | info
        "key_numbers": key_numbers,
        "charts": charts,
        "warnings": warnings,
        "details": details,
        "how_it_works": how,
    })


def kn(label: str, value: str, note: str = "") -> dict:
    return {"label": label, "value": value, "note": note}
