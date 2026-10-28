// Describes the settings form for each analysis method. The form is built from these specs, and the
// column guesses use the file's own column names so most examples need no clicking at all.
const few = (c) => c.n_unique >= 2 && c.n_unique <= 12;
const num = (c) => c.dtype === "number";

const GROUP = { key: "group_col", label: "Group column", help: "The column that says who was in control and who got the treatment.", kind: "column", filter: few, hint: /group|variant|arm|treat|bucket|cohort/i };
const CONTROL = { key: "control", label: "Control group is", kind: "value", of: "group_col", prefer: /^(control|a|baseline|base|0|false|no|old)$/i };
const TREATMENT = { key: "treatment", label: "Treatment group is", kind: "value", of: "group_col", prefer: /^(treat|b$|variant|test|1|true|yes|new)/i };
const ALPHA = { key: "alpha", label: "Significance level", kind: "select", options: [{ v: 0.05, t: "5% (standard)" }, { v: 0.01, t: "1% (stricter)" }, { v: 0.1, t: "10% (looser)" }], default: 0.05 };

export const SPECS = {
  ttest: { fields: [GROUP, CONTROL, TREATMENT,
    { key: "metric_col", label: "Metric to compare", help: "A number per user, for example revenue or minutes played.", kind: "column", filter: num, hint: /revenue|amount|value|metric|spend|time|minutes|score|outcome/i },
    ALPHA] },
  chisq: { fields: [GROUP, CONTROL, TREATMENT,
    { key: "outcome_col", label: "Outcome column", help: "Yes/no (or a category) per user, for example converted.", kind: "column", filter: (c) => c.n_unique >= 2 && c.n_unique <= 20, hint: /conver|click|purchase|retain|success|outcome|plan/i },
    { key: "success_value", label: "Value that counts as success", kind: "value", of: "outcome_col", optional: true, help: "Only needed for text outcomes with two values." },
    ALPHA] },
  multiple: { fields: [
    { key: "mode", label: "What do you have?", kind: "select", default: "from_data", options: [{ v: "from_data", t: "Raw data with several metrics" }, { v: "p_values", t: "A column of p-values I already have" }] },
    { ...GROUP, showIf: (p) => p.mode === "from_data" }, { ...CONTROL, showIf: (p) => p.mode === "from_data" }, { ...TREATMENT, showIf: (p) => p.mode === "from_data" },
    { key: "metric_cols", label: "Metric columns to test", help: "Every ticked metric gets its own test, then all are corrected together.", kind: "multicolumn", filter: num, hint: /metric/i, showIf: (p) => p.mode === "from_data" },
    { key: "label_col", label: "Label column", kind: "column", optional: true, showIf: (p) => p.mode === "p_values", hint: /name|label|metric|variant/i },
    { key: "p_col", label: "p-value column", kind: "column", filter: num, hint: /^p|p_?value/i, showIf: (p) => p.mode === "p_values" },
    ALPHA] },
  sequential: { fields: [GROUP, CONTROL, TREATMENT,
    { key: "metric_col", label: "Metric", help: "Yes/no (0/1) or a number per user.", kind: "column", filter: num, hint: /conver|revenue|metric|outcome|value/i },
    { key: "order_col", label: "Time or order column", help: "When each user arrived. Leave empty if rows are already in arrival order.", kind: "column", optional: true, hint: /time|date|order|day|ts/i },
    { key: "expected_effect", label: "Typical effect size to look for", help: "In the metric's own units. Leave empty to use 10% of the metric's standard deviation.", kind: "number", optional: true },
    ALPHA] },
  bayes: { fields: [GROUP, CONTROL, TREATMENT,
    { key: "metric_col", label: "Metric", kind: "column", filter: (c) => c.n_unique >= 2, hint: /conver|revenue|metric|outcome|value/i },
    { key: "metric_type", label: "Metric type", kind: "select", default: "auto", options: [{ v: "auto", t: "Detect automatically" }, { v: "binary", t: "Yes / no" }, { v: "numeric", t: "A number" }] },
    { key: "transform", label: "Transform (numbers only)", kind: "select", default: "none", options: [{ v: "none", t: "None" }, { v: "log1p", t: "Log(1 + value): for skewed money-like metrics" }] },
    { key: "min_effect", label: "Smallest worthwhile improvement", help: "Optional. In the metric's units (0.01 = one percentage point for a yes/no metric).", kind: "number", optional: true }] },
  psm: { fields: [
    { key: "treatment", label: "Treatment column (0 or 1)", kind: "column", filter: (c) => c.n_unique === 2, hint: /treat|exposed|program|group/i },
    { key: "outcome", label: "Outcome", kind: "column", filter: num, hint: /outcome|re78|earn|revenue|y$/i },
    { key: "covariates", label: "Characteristics to match on", help: "Measured before treatment, and plausibly related to both who is treated and the outcome.", kind: "multicolumn", filter: num },
    { key: "known_effect", label: "Known true effect (optional)", help: "Only if you have a benchmark, such as from a randomised trial. It is drawn as a reference line.", kind: "number", optional: true },
    { key: "refute", label: "Also run DoWhy's placebo refutation (slower)", kind: "bool", default: false },
    { key: "caliper", label: "Matching tolerance (caliper)", help: "Maximum gap between matched units, in standard deviations of the score. 0.2 is the usual choice.", kind: "number", default: 0.2 }] },
  did: { fields: [
    { key: "outcome", label: "Outcome", kind: "column", filter: num, hint: /outcome|rate|value|y$/i },
    { ...GROUP, label: "Group column", help: "Which units were affected and which were not.", filter: (c) => c.n_unique >= 2 },
    { key: "treated_value", label: "Treated group is", kind: "value", of: "group_col", prefer: /treat|california|1|true|yes/i },
    { key: "time_col", label: "Time column (numeric)", kind: "column", filter: num, hint: /time|period|year|quarter|month|week|day/i },
    { key: "treat_time", label: "First period after the change", help: "Rows at or after this value count as 'after'.", kind: "number" },
    { key: "unit_col", label: "Unit column", help: "Which row belongs to which state, store, user, etc. Recommended: it gives honest uncertainty.", kind: "column", optional: true, hint: /unit|state|store|id|region|city/i }] },
  synth: { fields: [
    { key: "outcome", label: "Outcome", kind: "column", filter: num, hint: /outcome|rate|prison|value|y$/i },
    { key: "unit_col", label: "Unit column", help: "The things being compared: states, stores, countries.", kind: "column", filter: (c) => c.n_unique >= 4, hint: /unit|state|store|region|country/i },
    { key: "time_col", label: "Time column (numeric)", kind: "column", filter: num, hint: /time|period|year|month|week/i },
    { key: "treated_unit", label: "The treated unit", kind: "value", of: "unit_col" },
    { key: "treat_time", label: "First period after the change", kind: "number" }] },
};
