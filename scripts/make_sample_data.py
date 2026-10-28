"""Write the sample data sets used by the demo and the documentation into sample_data/.

Run from the repository root:  python scripts/make_sample_data.py
Real data sets come from the `causaldata` package (installed with the requirements).
"""
from pathlib import Path

import pandas as pd
from causaldata import cps_mixtape, nsw_mixtape, organ_donations, texas

from expplat import simulate

OUT = Path(__file__).resolve().parents[1] / "sample_data"
OUT.mkdir(exist_ok=True)

# --- simulated experiments (real effect planted: +1.2 percentage points conversion)
simulate.simulate_ab(n=20000, base_rate=0.10, lift=0.012, seed=2026).to_csv(OUT / "sim_ab_test.csv", index=False)
simulate.simulate_multi_metrics(seed=2026).to_csv(OUT / "sim_multiple_metrics.csv", index=False)
simulate.simulate_observational(seed=2026).to_csv(OUT / "sim_observational.csv", index=False)
simulate.simulate_did(seed=2026).to_csv(OUT / "sim_did_panel.csv", index=False)
simulate.simulate_synth(seed=2026).to_csv(OUT / "sim_synthetic_control.csv", index=False)

# --- LaLonde job training: randomised treated group + non-experimental (CPS) comparison group
nsw = nsw_mixtape.load_pandas().data
cps = cps_mixtape.load_pandas().data
lalonde = pd.concat([nsw[nsw.treat == 1], cps], ignore_index=True)
lalonde.to_csv(OUT / "lalonde_nsw_cps.csv", index=False)
nsw.to_csv(OUT / "lalonde_nsw_experiment.csv", index=False)

# --- organ donations (California changed its donor-registration rules in 2011 Q3)
od = organ_donations.load_pandas().data
od["group"] = (od["State"] == "California").map({True: "California", False: "Other states"})
od.to_csv(OUT / "organ_donations_did.csv", index=False)

# --- Texas prison expansion (synthetic control); treated: Texas, from 1993
tx = texas.load_pandas().data[["state", "year", "bmprison", "income", "ur", "poverty", "black"]]
tx.to_csv(OUT / "texas_prison_synth.csv", index=False)
for p in sorted(OUT.glob("*.csv")):
    print(p.name, len(pd.read_csv(p)), "rows")
