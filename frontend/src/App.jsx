import { useEffect, useState } from "react";
import { getCatalog } from "./api.js";
import AnalyzePanel from "./components/AnalyzePanel.jsx";
import PowerPanel from "./components/PowerPanel.jsx";

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [catalog, setCatalog] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { getCatalog().then(setCatalog).catch((e) => setErr(e.message)); }, []);

  return (
    <>
      <header className="top">
        <div className="top-in">
          <div className="brand">
            <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="var(--accent)" /><path d="M9 22l5-7 4 4 5-8" fill="none" stroke="white" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
            <div><strong>Experiment Lab</strong><span>Design tests, analyse them, estimate cause and effect</span></div>
          </div>
          <nav aria-label="Main">
            <button className={tab === "plan" ? "on" : ""} onClick={() => setTab("plan")}>Plan a test</button>
            <button className={tab === "analyze" ? "on" : ""} onClick={() => setTab("analyze")}>Analyse data</button>
          </nav>
        </div>
      </header>
      <main className="page">
        {tab === "plan" && <PowerPanel />}
        {tab === "analyze" && (catalog ? <AnalyzePanel catalog={catalog} /> : err ? <div className="alert">Could not reach the analysis service: {err}</div> : <div className="card empty">Loading…</div>)}
      </main>
      <footer className="foot">
        Results are statistical estimates, not guarantees. Causal methods rely on assumptions the data cannot prove; read the cautions shown with each result.
      </footer>
    </>
  );
}
