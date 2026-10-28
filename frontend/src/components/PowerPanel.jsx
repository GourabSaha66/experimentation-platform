import { useEffect, useRef, useState } from "react";
import { postPower } from "../api.js";
import ResultView from "./ResultView.jsx";

const DEFAULTS = { kind: "proportions", solve_for: "sample_size", baseline: 10, mde: 10, mde_type: "relative", sd: 20, n_per_group: 5000,
  alpha: 0.05, power: 0.8, ratio: 1, alternative: "two-sided", daily_users: "" };

function buildBody(s) {
  const body = { kind: s.kind, solve_for: s.solve_for, alpha: Number(s.alpha), power: Number(s.power), ratio: Number(s.ratio), alternative: s.alternative };
  if (s.kind === "proportions") {
    body.baseline_rate = Number(s.baseline) / 100;
    if (s.solve_for !== "mde") { body.mde_type = s.mde_type; body.mde = s.mde_type === "relative" ? Number(s.mde) / 100 : Number(s.mde) / 100; }
  } else {
    body.sd = Number(s.sd);
    if (s.solve_for !== "mde") body.mde = Number(s.mde);
  }
  if (s.solve_for !== "sample_size") body.n_per_group = Math.round(Number(s.n_per_group));
  if (s.daily_users !== "" && Number(s.daily_users) > 0) body.daily_users = Math.round(Number(s.daily_users));
  return body;
}

export default function PowerPanel() {
  const [s, setS] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const seq = useRef(0);
  const set = (k, v) => setS((p) => ({ ...p, [k]: v }));

  useEffect(() => {
    const id = ++seq.current;
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        const r = await postPower(buildBody(s));
        if (id === seq.current) { setResult(r); setError(""); }
      } catch (e) {
        if (id === seq.current) { setError(e.message); }
      } finally { if (id === seq.current) setBusy(false); }
    }, 350);
    return () => clearTimeout(t);
  }, [s]);

  const prop = s.kind === "proportions";
  const solve = s.solve_for;
  return (
    <div className="split">
      <aside className="card side">
        <h2>Plan a test</h2>
        <p className="muted">How many users do you need, how much power do you have, or how small an effect can you detect? Results update as you type.</p>

        <div className="seg" role="group" aria-label="Metric type">
          <button className={prop ? "on" : ""} onClick={() => set("kind", "proportions")}>Conversion rate</button>
          <button className={!prop ? "on" : ""} onClick={() => set("kind", "means")}>Average value</button>
        </div>

        <label className="field"><span className="lbl">I want to find</span>
          <select value={solve} onChange={(e) => set("solve_for", e.target.value)}>
            <option value="sample_size">How many users I need</option>
            <option value="power">The power I will have</option>
            <option value="mde">The smallest effect I can detect</option>
          </select></label>

        {prop ? (
          <label className="field"><span className="lbl">Current conversion rate (%)</span>
            <input type="number" step="any" value={s.baseline} onChange={(e) => set("baseline", e.target.value)} />
            <small>What fraction of users do the thing today?</small></label>
        ) : (
          <label className="field"><span className="lbl">Standard deviation of the metric</span>
            <input type="number" step="any" value={s.sd} onChange={(e) => set("sd", e.target.value)} />
            <small>How much the metric varies from user to user. Take it from past data.</small></label>
        )}

        {solve !== "mde" && (
          <div className="row">
            <label className="field grow"><span className="lbl">Smallest effect worth detecting</span>
              <input type="number" step="any" value={s.mde} onChange={(e) => set("mde", e.target.value)} /></label>
            {prop && (
              <label className="field"><span className="lbl">as</span>
                <select value={s.mde_type} onChange={(e) => set("mde_type", e.target.value)}>
                  <option value="relative">% relative</option>
                  <option value="absolute">pp absolute</option>
                </select></label>
            )}
          </div>
        )}

        {solve !== "sample_size" && (
          <label className="field"><span className="lbl">Users per group (control)</span>
            <input type="number" value={s.n_per_group} onChange={(e) => set("n_per_group", e.target.value)} /></label>
        )}

        <div className="row">
          <label className="field grow"><span className="lbl">Significance level</span>
            <select value={s.alpha} onChange={(e) => set("alpha", e.target.value)}>
              <option value="0.05">5%</option><option value="0.01">1%</option><option value="0.1">10%</option></select></label>
          {solve !== "power" && (
            <label className="field grow"><span className="lbl">Power</span>
              <select value={s.power} onChange={(e) => set("power", e.target.value)}>
                <option value="0.8">80%</option><option value="0.9">90%</option><option value="0.95">95%</option></select></label>
          )}
        </div>
        <div className="row">
          <label className="field grow"><span className="lbl">Treatment / control size</span>
            <input type="number" step="any" value={s.ratio} onChange={(e) => set("ratio", e.target.value)} /></label>
          <label className="field grow"><span className="lbl">Users per day <em>optional</em></span>
            <input type="number" value={s.daily_users} onChange={(e) => set("daily_users", e.target.value)} placeholder="e.g. 2000" /></label>
        </div>
        <label className="field"><span className="lbl">Test type</span>
          <select value={s.alternative} onChange={(e) => set("alternative", e.target.value)}>
            <option value="two-sided">Two-sided: could be better or worse</option>
            <option value="larger">One-sided: only care if it is better</option></select></label>
      </aside>

      <div className="main">
        {error && <div className="alert" role="alert">{error}</div>}
        {result ? <div className={busy ? "dim" : ""}><ResultView result={result} /></div> : !error && <div className="card empty">Calculating…</div>}
      </div>
    </div>
  );
}
