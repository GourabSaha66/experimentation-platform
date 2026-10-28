import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from expplat.main import app
from expplat.samples import EXAMPLES

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[1] / "sample_data"
FAST = [e for e in EXAMPLES if e["method"] in ("ttest", "chisq", "multiple", "sequential", "did", "synth")]


def test_health_and_catalog():
    assert client.get("/api/health").json()["status"] == "ok"
    cat = client.get("/api/methods").json()
    assert {m["id"] for m in cat["methods"]} == {"ttest", "chisq", "multiple", "sequential", "bayes", "psm", "did", "synth"}
    for e in cat["examples"]:
        assert (SAMPLES / e["file"]).exists()


def test_power_endpoint_and_validation():
    r = client.post("/api/power", json={"kind": "proportions", "solve_for": "sample_size", "baseline_rate": 0.1, "mde": 0.1})
    assert r.status_code == 200 and r.json()["details"]["n_control"] > 10000
    assert client.post("/api/power", json={"kind": "means", "solve_for": "sample_size", "mde": 1}).status_code == 422


@pytest.mark.parametrize("ex", FAST, ids=[e["id"] for e in FAST])
def test_every_fast_example_runs_through_the_api(ex):
    with open(SAMPLES / ex["file"], "rb") as f:
        r = client.post(f"/api/analyze/{ex['method']}", files={"file": (ex["file"], f, "text/csv")}, data={"params": json.dumps(ex["params"])})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["conclusion"] and body["charts"] and body["key_numbers"]
    assert body["method"] == ex["method"]


def test_inspect_describes_columns():
    with open(SAMPLES / "sim_ab_test.csv", "rb") as f:
        r = client.post("/api/inspect", files={"file": ("x.csv", f, "text/csv")})
    cols = {c["name"]: c for c in r.json()["columns"]}
    assert r.json()["rows"] == 20000 and set(cols["group"]["values"]) == {"control", "treatment"}


def test_helpful_errors():
    with open(SAMPLES / "sim_ab_test.csv", "rb") as f:
        r = client.post("/api/analyze/ttest", files={"file": ("x.csv", f, "text/csv")},
                        data={"params": json.dumps({"group_col": "nope", "metric_col": "revenue", "control": "a", "treatment": "b"})})
    assert r.status_code == 422 and "not in your file" in r.json()["detail"]
    r = client.post("/api/analyze/ttest", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}, data={"params": "{}"})
    assert r.status_code == 422 and "missing or invalid" in r.json()["detail"]
    r = client.post("/api/analyze/magic", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}, data={"params": "{}"})
    assert r.status_code == 404
    r = client.post("/api/analyze/ttest", files={"file": ("x.csv", b"", "text/csv")}, data={"params": "{}"})
    assert r.status_code == 422


def test_sample_download_is_confined_to_sample_dir():
    assert client.get("/api/samples/sim_ab_test.csv").status_code == 200
    assert client.get("/api/samples/..%2Frequirements.txt").status_code in (404, 422)
