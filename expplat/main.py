"""FastAPI service: power calculator + every analysis method behind one small JSON API."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import __version__
from .ab_tests import analyze_chisq, analyze_multiple, analyze_ttest
from .bayes import analyze_bayes
from .causal import analyze_did, analyze_psm, analyze_synth
from .common import UserError, clean
from .power import plan
from .samples import EXAMPLES, METHODS
from .sequential import analyze_sequential

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
DIST_DIR = ROOT / "frontend" / "dist"
MAX_BYTES = 25 * 1024 * 1024
MAX_ROWS = 500_000

app = FastAPI(title="Experimentation & Causal Inference Platform", version=__version__,
              description="Design A/B tests, analyse them correctly, and estimate cause and effect when you cannot randomise.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# The heavy methods (MCMC, bootstraps) are run one at a time so a small server is not overloaded.
_heavy = threading.Semaphore(1)


@app.exception_handler(UserError)
async def _user_error(_: Request, exc: UserError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ------------------------------------------------------------------ request models
class PowerRequest(BaseModel):
    kind: Literal["means", "proportions"] = "proportions"
    solve_for: Literal["sample_size", "power", "mde"] = "sample_size"
    alpha: float = Field(0.05, gt=0, lt=1)
    power: float | None = Field(0.8, gt=0, lt=1)
    ratio: float = Field(1.0, gt=0, description="treatment group size / control group size")
    alternative: Literal["two-sided", "larger"] = "two-sided"
    baseline_rate: float | None = Field(None, gt=0, lt=1)
    sd: float | None = Field(None, gt=0)
    mde: float | None = Field(None, description="smallest effect worth detecting")
    mde_type: Literal["relative", "absolute"] = "relative"
    n_per_group: int | None = Field(None, gt=1)
    daily_users: int | None = Field(None, gt=0)


class _Groups(BaseModel):
    group_col: str
    control: str
    treatment: str
    alpha: float = Field(0.05, gt=0, lt=1)


class TTestParams(_Groups):
    metric_col: str
    expected_ratio: float = 1.0


class ChisqParams(_Groups):
    outcome_col: str
    success_value: str | None = None
    expected_ratio: float = 1.0


class MultipleParams(BaseModel):
    mode: Literal["from_data", "p_values"] = "from_data"
    alpha: float = Field(0.05, gt=0, lt=1)
    group_col: str | None = None
    control: str | None = None
    treatment: str | None = None
    metric_cols: list[str] | None = None
    label_col: str | None = None
    p_col: str | None = None


class SequentialParams(_Groups):
    metric_col: str
    order_col: str | None = None
    expected_effect: float | None = Field(None, gt=0)
    n_looks: int = Field(60, ge=5, le=300)


class BayesParams(BaseModel):
    group_col: str
    metric_col: str
    control: str
    treatment: str
    metric_type: Literal["auto", "binary", "numeric"] = "auto"
    success_value: str | None = None
    min_effect: float = 0.0
    transform: Literal["none", "log1p"] = "none"
    draws: int = Field(2000, ge=500, le=4000)
    seed: int = 42
    expected_ratio: float = 1.0


class PSMParams(BaseModel):
    treatment: str
    outcome: str
    covariates: list[str]
    caliper: float = Field(0.2, gt=0, le=2)
    n_boot: int = Field(200, ge=50, le=500)
    crosscheck: bool = True
    refute: bool = False
    known_effect: float | None = None
    alpha: float = Field(0.05, gt=0, lt=1)


class DIDParams(BaseModel):
    outcome: str
    group_col: str
    treated_value: str
    time_col: str
    treat_time: float
    unit_col: str | None = None
    alpha: float = Field(0.05, gt=0, lt=1)


class SynthParams(BaseModel):
    outcome: str
    unit_col: str
    time_col: str
    treated_unit: str
    treat_time: float


# method id -> (params model, runner, is_heavy)
REGISTRY = {
    "ttest": (TTestParams, lambda df, p: analyze_ttest(df, **p.model_dump()), False),
    "chisq": (ChisqParams, lambda df, p: analyze_chisq(df, **p.model_dump()), False),
    "multiple": (MultipleParams, lambda df, p: analyze_multiple(df, **p.model_dump()), False),
    "sequential": (SequentialParams, lambda df, p: analyze_sequential(df, **p.model_dump()), False),
    "bayes": (BayesParams, lambda df, p: analyze_bayes(df, **p.model_dump()), True),
    "psm": (PSMParams, lambda df, p: analyze_psm(df, **p.model_dump()), True),
    "did": (DIDParams, lambda df, p: analyze_did(df, **p.model_dump()), False),
    "synth": (SynthParams, lambda df, p: analyze_synth(df, **p.model_dump()), False),
}


# ------------------------------------------------------------------ helpers
def _read_csv(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise UserError(f"That file is larger than {MAX_BYTES // 1024 // 1024} MB. Use a smaller extract.")
    if not raw.strip():
        raise UserError("The file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS + 1, sep=None, engine="python", encoding_errors="replace")
    except Exception as e:
        raise UserError(f"Could not read the file as a CSV table ({str(e)[:120]}).") from e
    if len(df) > MAX_ROWS:
        raise UserError(f"The file has more than {MAX_ROWS:,} rows. Use a sample or aggregate it first.")
    if df.shape[1] < 2:
        raise UserError("The file needs at least two columns. Is it comma-separated?")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/methods")
def methods():
    return {"methods": METHODS, "examples": EXAMPLES}


@app.get("/api/samples/{name}")
def sample_file(name: str):
    path = (SAMPLE_DIR / name).resolve()
    if path.parent != SAMPLE_DIR.resolve() or path.suffix != ".csv" or not path.exists():
        raise HTTPException(404, "No such sample file.")
    return FileResponse(path, media_type="text/csv", filename=name)


@app.post("/api/power")
def power(req: PowerRequest):
    return clean(plan(**req.model_dump()))


@app.post("/api/inspect")
def inspect(file: UploadFile = File(...)):
    """Read an uploaded CSV and describe its columns so the form can offer sensible choices."""
    df = _read_csv(file)
    cols = []
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        cols.append({"name": c, "dtype": "number" if pd.api.types.is_numeric_dtype(s) else "text", "n_unique": nun,
                     "missing": int(s.isna().sum()),
                     "values": sorted(str(v) for v in s.dropna().unique()) if nun <= 200 else [],
                     "examples": [str(v) for v in s.dropna().head(3)]})
    return clean({"rows": len(df), "columns": cols, "preview": df.head(8).astype(object).where(df.head(8).notna(), None).to_dict(orient="records")})


@app.post("/api/analyze/{method}")
def analyze(method: str, file: UploadFile = File(...), params: str = Form("{}")):
    if method not in REGISTRY:
        raise HTTPException(404, f"Unknown method '{method}'. Choose one of: {', '.join(REGISTRY)}.")
    model, runner, heavy = REGISTRY[method]
    try:
        p = model(**json.loads(params or "{}"))
    except json.JSONDecodeError as e:
        raise UserError("The settings could not be read.") from e
    except ValidationError as e:
        msgs = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        raise UserError("Some settings are missing or invalid: " + "; ".join(msgs)) from e
    df = _read_csv(file)
    if heavy:
        with _heavy:
            return runner(df, p)
    return runner(df, p)


if DIST_DIR.exists():  # the built React app, when present (production image)
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")
