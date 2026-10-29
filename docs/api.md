# API reference

The service is a small JSON API. Interactive documentation (generated from the code) is always available at `/docs` (Swagger UI) and `/openapi.json` while the server runs.

Base URL: `http://127.0.0.1:8000` locally.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/methods` | Method catalogue and the ready-made examples |
| POST | `/api/power` | Sample size, power or minimum detectable effect (JSON body) |
| POST | `/api/inspect` | Describe the columns of an uploaded CSV (multipart `file`) |
| POST | `/api/analyze/{method}` | Run an analysis (multipart `file` + `params` JSON string) |
| GET | `/api/samples/{name}` | Download a sample CSV |

`{method}` is one of `ttest`, `chisq`, `multiple`, `sequential`, `bayes`, `psm`, `did`, `synth`.

## Limits

CSV files up to 25 MB and 500,000 rows. Bayesian and matching analyses run one at a time (a semaphore), so a small server is never overloaded; other requests are not blocked.

## Power calculator: `POST /api/power`

```json
{
  "kind": "proportions",          // "proportions" or "means"
  "solve_for": "sample_size",     // "sample_size", "power" or "mde"
  "alpha": 0.05, "power": 0.8, "ratio": 1.0,
  "alternative": "two-sided",     // or "larger" (one-sided)
  "baseline_rate": 0.10,          // proportions only
  "sd": null,                     // means only
  "mde": 0.10, "mde_type": "relative",   // "relative" or "absolute"
  "n_per_group": null,            // needed when solving for power or mde
  "daily_users": 2000             // optional, adds "days needed"
}
```

```bash
curl -s localhost:8000/api/power -H "Content-Type: application/json" \
  -d '{"kind":"proportions","solve_for":"sample_size","baseline_rate":0.1,"mde":0.1}'
```

## Analyses: `POST /api/analyze/{method}`

Send `multipart/form-data` with a CSV in `file` and the settings as a JSON string in `params`.

```bash
curl -s localhost:8000/api/analyze/ttest \
  -F file=@sample_data/sim_ab_test.csv \
  -F 'params={"group_col":"group","metric_col":"revenue","control":"control","treatment":"treatment"}'
```

### Settings per method

| Method | Required settings | Optional settings |
|---|---|---|
| `ttest` | `group_col`, `control`, `treatment`, `metric_col` | `alpha` (0.05), `expected_ratio` (1.0, for the sample-ratio check) |
| `chisq` | `group_col`, `control`, `treatment`, `outcome_col` | `alpha`, `success_value`, `expected_ratio` |
| `multiple` | `mode` = `from_data`: `group_col`, `control`, `treatment`, `metric_cols[]`; `mode` = `p_values`: `p_col` | `alpha`, `label_col` |
| `sequential` | `group_col`, `control`, `treatment`, `metric_col` | `order_col`, `expected_effect`, `alpha`, `n_looks` (60) |
| `bayes` | `group_col`, `control`, `treatment`, `metric_col` | `metric_type` (`auto`/`binary`/`numeric`), `success_value`, `transform` (`none`/`log1p`), `min_effect`, `draws` (2000) |
| `psm` | `treatment` (0/1), `outcome`, `covariates[]` | `caliper` (0.2), `n_boot` (200), `crosscheck` (true), `refute` (false), `known_effect`, `alpha` |
| `did` | `outcome`, `group_col`, `treated_value`, `time_col` (numeric), `treat_time` | `unit_col`, `alpha` |
| `synth` | `outcome`, `unit_col`, `time_col` (numeric), `treated_unit`, `treat_time` | none |

### Response (the same shape for every method)

```json
{
  "method": "ttest",
  "title": "Two-group comparison of means (Welch t-test)",
  "conclusion": "Plain-English paragraph ...",
  "verdict": "significant",          // significant | not_significant | inconclusive | info
  "key_numbers": [{"label": "Difference", "value": "0.246", "note": "95% CI ..."}],
  "charts": [ ... ],                 // see below
  "warnings": ["Sample ratio mismatch: ..."],
  "details": { ... },                // the raw numbers behind the text
  "how_it_works": "Short explanation of the method"
}
```

### Chart specifications

Charts are described as data, and the React app draws them. Kinds:

* `line`: `series[{name, points[{x, y, lo?, hi?}], dashed?, fill?, muted?, strong?}]`, optional `vlines`, `hlines`, `log_y`.
* `bar`: `categories[]`, `series[{name, values[]}]`, optional `stacked`, `horizontal`, `hline`, `marker`.
* `forest`: `rows[{label, estimate, lo, hi}]` and a reference value `ref`.

### Errors

Problems with the data or settings return HTTP 422 with a message written for the user:

```json
{"detail": "These columns are not in your file: 'nope'. Columns found: 'user_id', 'timestamp', 'group', ..."}
```

Other cases: 404 for an unknown method or sample file.

## The statistics behind each endpoint

| Method | What is computed |
|---|---|
| Power | Normal-approximation power for two groups (statsmodels `TTestIndPower`, `NormalIndPower` with Cohen's h). |
| `ttest` | Welch t-test, interval for the difference, Cohen's d, approximate interval for the relative change (delta method), sample-ratio-mismatch check. |
| `chisq` | 2×2 chi-square without continuity correction, Newcombe interval for the difference, interval for the relative lift, Fisher's exact test when expected counts are below 5; general r×c table with Cramér's V. |
| `multiple` | Welch tests per metric (or supplied p-values), then Bonferroni, Holm and Benjamini-Hochberg adjusted p-values. |
| `sequential` | Mixture sequential probability ratio test: always-valid p-value and confidence sequence for a difference in means, evaluated at up to 300 looks. |
| `bayes` | PyMC: Beta-Binomial for yes/no metrics; Normal model on sufficient statistics for numeric metrics. Reports P(B beats A), credible interval, expected loss, R-hat. |
| `psm` | Logistic propensity score, 1:1 nearest-neighbour matching with replacement inside a caliper, bootstrap interval, covariate balance, overlap plot. Cross-checks: DoWhy matching (and optional placebo refutation) and EconML `LinearDRLearner` on the region with overlap. |
| `did` | Two-period difference-in-differences by regression with unit-clustered standard errors (t distribution), pre-trend test, placebo-in-space test, event-study style gap chart. |
| `synth` | Non-negative weights that sum to one, fitted to the pre-change outcome path; placebo-in-space p-value from the post/pre error ratio. |
