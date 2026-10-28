import { useEffect, useMemo, useRef, useState } from "react";
import { analyze, fetchSample, inspectFile } from "../api.js";
import ResultView from "./ResultView.jsx";
import SettingsForm, { cleanParams, guessParams, missingRequired } from "./SettingsForm.jsx";

const SLOW = { bayes: "Running Bayesian sampling. This usually takes 10 to 40 seconds.", psm: "Matching, bootstrapping and cross-checking with DoWhy and EconML. This can take up to a minute." };

export default function AnalyzePanel({ catalog }) {
  const [method, setMethod] = useState("ttest");
  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [params, setParams] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const pendingParams = useRef(null);
  const resultRef = useRef(null);

  const methods = catalog.methods;
  const examples = useMemo(() => catalog.examples.filter((e) => e.method === method), [catalog, method]);
  const groups = useMemo(() => [...new Set(methods.map((m) => m.group))], [methods]);

  // when the method or the file changes, re-guess the settings (unless an example supplied them)
  useEffect(() => {
    if (!info) return;
    if (pendingParams.current) { setParams({ ...guessParams(method, info.columns), ...pendingParams.current }); pendingParams.current = null; }
    else setParams(guessParams(method, info.columns));
    setResult(null);
  }, [method, info]);

  async function loadFile(f, exampleParams) {
    setError(""); setResult(null); setBusy(true);
    try {
      const i = await inspectFile(f);
      pendingParams.current = exampleParams || null;
      setFile(f); setInfo(i);
    } catch (e) { setError(e.message); setFile(null); setInfo(null); }
    finally { setBusy(false); }
  }

  async function useExample(ex) {
    setError(""); setBusy(true);
    try { const f = await fetchSample(ex.file); setBusy(false); await loadFile(f, ex.params); }
    catch (e) { setError(e.message); setBusy(false); }
  }

  async function run() {
    setError(""); setResult(null); setBusy(true);
    try {
      const r = await analyze(method, file, cleanParams(method, params));
      setResult(r);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  const missing = info ? missingRequired(method, params) : ["a data file"];
  const cur = methods.find((m) => m.id === method);

  return (
    <div className="stack">
      <section className="card">
        <h2><span className="step">1</span> What do you want to find out?</h2>
        {groups.map((g) => (
          <div key={g} className="method-group">
            <h3>{g === "Experiments" ? "I ran an experiment" : "I could not run an experiment"}</h3>
            <div className="method-grid">
              {methods.filter((m) => m.group === g).map((m) => (
                <button key={m.id} className={"method" + (m.id === method ? " on" : "")} onClick={() => setMethod(m.id)} aria-pressed={m.id === method}>
                  <strong>{m.name}</strong><span>{m.tagline}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
        <p className="muted small">{cur?.needs}</p>
      </section>

      <section className="card">
        <h2><span className="step">2</span> Add your data</h2>
        <div className={"drop" + (drag ? " over" : "")}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) loadFile(f); }}>
          <p>{file ? <><strong>{file.name}</strong>: {info?.rows.toLocaleString()} rows, {info?.columns.length} columns</> : "Drop a CSV file here, or"}</p>
          <label className="btn">Choose a file<input type="file" accept=".csv,text/csv,.txt,.tsv" hidden onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])} /></label>
        </div>
        {examples.length > 0 && (
          <div className="examples">
            <span className="muted small">Or try an example:</span>
            {examples.map((ex) => (
              <button key={ex.id} className="chip" onClick={() => useExample(ex)} title={ex.about}>{ex.label}</button>
            ))}
          </div>
        )}
        {info && (
          <details className="preview">
            <summary>Preview of your data</summary>
            <div className="table-wrap">
              <table>
                <thead><tr>{info.columns.map((c) => <th key={c.name}>{c.name}</th>)}</tr></thead>
                <tbody>{info.preview.map((r, i) => <tr key={i}>{info.columns.map((c) => <td key={c.name}>{String(r[c.name] ?? "")}</td>)}</tr>)}</tbody>
              </table>
            </div>
          </details>
        )}
      </section>

      {info && (
        <section className="card">
          <h2><span className="step">3</span> Check the settings</h2>
          {examples.find((e) => file && e.file === file.name)?.about && <p className="muted">{examples.find((e) => e.file === file.name).about}</p>}
          <SettingsForm method={method} columns={info.columns} params={params} setParams={setParams} />
          <div className="run-row">
            <button className="btn primary" disabled={busy || missing.length > 0} onClick={run}>{busy ? "Working…" : "Run analysis"}</button>
            {missing.length > 0 && <span className="muted small">Still needed: {missing.join(", ")}</span>}
          </div>
          {busy && SLOW[method] && <p className="muted small" role="status">{SLOW[method]}</p>}
        </section>
      )}

      {error && <div className="alert" role="alert">{error}</div>}
      <div ref={resultRef}>{result && <ResultView result={result} />}</div>
    </div>
  );
}
