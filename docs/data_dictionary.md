# Sample data

All files live in `sample_data/` and are produced by `python scripts/make_sample_data.py`. The simulated files use fixed seeds, so they are identical on every machine.

## Simulated (generated in code, `expplat/simulate.py`)

| File | Rows | Columns | Built-in truth |
|---|---|---|---|
| `sim_ab_test.csv` | 20,000 | `user_id`, `timestamp` (28 days, sorted), `group` (control/treatment), `converted` (0/1), `revenue` (0 unless converted, then log-normal) | Conversion 10% in control and 11.2% in treatment (+1.2 percentage points) |
| `sim_multiple_metrics.csv` | 6,000 | `group`, `metric_01` ... `metric_12` (standard normal noise) | Metrics 01-03 are shifted by 0.12 sd in treatment; 04-12 have no effect |
| `sim_observational.csv` | 4,000 | `age`, `income`, `prior_score`, `treated` (0/1), `outcome` | True treatment effect 2.0; treatment is more likely for people who would score higher anyway |
| `sim_did_panel.csv` | 480 | `unit`, `period` (1-12), `group` (treated/control), `outcome` | 8 treated and 32 control units; true effect 3.0 from period 7 |
| `sim_synthetic_control.csv` | 780 | `unit` (25 donors + `treated_unit`), `period` (1-30), `outcome` | Treated unit is a mixture of three donors; true effect -4.0 from period 21 |

## Real data (from the `causaldata` Python package)

| File | Rows | Source and meaning |
|---|---|---|
| `lalonde_nsw_experiment.csv` | 445 | National Supported Work (NSW) job-training experiment, Dehejia-Wahba sample: 185 treated and 260 randomised controls. `re78` is 1978 earnings. The randomised difference (about $1,794) is the benchmark answer. |
| `lalonde_nsw_cps.csv` | 16,177 | The 185 treated men from the experiment plus 15,992 comparison men from the Current Population Survey (CPS). A non-experimental setting where naive comparison is badly biased. |
| `organ_donations_did.csv` | 162 | State-by-quarter organ-donor registration rates (27 states, Q4 2010 to Q1 2012). California changed its registration rules in Q3 2011 (`Quarter_Num` 4). `group` marks California. |
| `texas_prison_synth.csv` | 816 | State-by-year panel, 1985-2000 (51 states). `bmprison` is the number of Black men in prison. Texas expanded prison capacity from 1993. |

The `causaldata` package (Nick Huntington-Klein, MIT licence) bundles data sets used in the textbooks *Causal Inference: The Mixtape* and *The Effect*. See https://github.com/NickCH-K/causaldata for the original sources and citations.

The assignment's other options (Cookie Cats, Criteo/Hillstrom, IHDP, Card and Krueger, California Proposition 99) are not used here: they need a Kaggle or external download that was not available, so the equivalent roles are filled by the simulated A/B data, LaLonde, the organ-donation panel and the Texas panel.
