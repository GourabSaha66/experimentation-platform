"""Drive the real web app with a headless browser: click every built-in example, check that a
conclusion and charts appear, and save a screenshot of each result to docs/screenshots/.

Start the server first (uvicorn expplat.main:app --port 8000), then:  python scripts/e2e_screenshots.py
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [("ttest", "Compare averages"), ("chisq", "Compare rates"), ("multiple", "Many metrics or variants"), ("sequential", "Peek safely"),
           ("bayes", "Probability B beats A"), ("psm", "Match similar units"), ("did", "Before and after vs. a comparison group"),
           ("synth", "Build a synthetic comparison")]
EXAMPLE = {"ttest": 0, "chisq": 0, "multiple": 0, "sequential": 0, "bayes": 0, "psm": 0, "did": 0, "synth": 0}
failures = []

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

    # a sticky header repeats itself in full-page screenshots, so turn the stickiness off here
    page.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='.top{position:static!important}.side{position:static!important}';document.head.appendChild(s)})")
    # power calculator
    page.goto(BASE)
    page.get_by_role("button", name="Plan a test").click()
    page.wait_for_selector(".conclusion", timeout=20000)
    txt = page.inner_text(".conclusion")
    print("power:", txt[:110])
    page.screenshot(path=str(OUT / "01_plan_a_test.png"), full_page=True)

    page.get_by_role("button", name="Analyse data").click()
    for i, (mid, name) in enumerate(METHODS, start=2):
        page.get_by_role("button", name=name, exact=False).first.click()
        chips = page.locator(".chip")
        chips.nth(EXAMPLE[mid]).click()
        page.get_by_role("button", name="Run analysis").wait_for(timeout=20000)
        page.get_by_role("button", name="Run analysis").click()
        try:
            page.wait_for_selector(".result .conclusion", timeout=180000)
            page.wait_for_timeout(300)
            n_charts = page.locator(".chart-card").count()
            print(f"{mid}: ok, {n_charts} charts, '{page.inner_text('.result .conclusion')[:90]}...'")
            if n_charts == 0:
                failures.append(f"{mid}: no charts")
            page.screenshot(path=str(OUT / f"{i:02d}_{mid}.png"), full_page=True)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mid}: {e}")
            page.screenshot(path=str(OUT / f"{i:02d}_{mid}_FAILED.png"), full_page=True)
        # start the next method from a clean result
        page.reload()
        page.get_by_role("button", name="Analyse data").click()
    b.close()
    if errors:
        failures.append("browser errors: " + "; ".join(errors[:3]))

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
