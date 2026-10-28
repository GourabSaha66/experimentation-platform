import { useState } from "react";
import Chart from "./Charts.jsx";

const VERDICT = {
  significant: { text: "Evidence of an effect", cls: "good" },
  not_significant: { text: "No clear evidence of an effect", cls: "neutral" },
  inconclusive: { text: "Not decided yet", cls: "warn" },
  info: { text: "Result", cls: "info" },
};

export default function ResultView({ result, onReset }) {
  const [copied, setCopied] = useState(false);
  const [showTech, setShowTech] = useState(false);
  const v = VERDICT[result.verdict] || VERDICT.info;

  const copy = async () => {
    try { await navigator.clipboard.writeText(result.conclusion); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch (_) { /* clipboard blocked */ }
  };
  const download = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${result.method}-result.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <section className="result" aria-live="polite">
      <header className="result-head">
        <div>
          <p className="eyebrow">{result.title}</p>
          <span className={"badge " + v.cls}>{v.text}</span>
        </div>
        <div className="actions">
          <button className="btn ghost" onClick={copy}>{copied ? "Copied" : "Copy conclusion"}</button>
          <button className="btn ghost" onClick={download}>Download JSON</button>
          {onReset && <button className="btn ghost" onClick={onReset}>Start over</button>}
        </div>
      </header>

      <p className="conclusion">{result.conclusion}</p>

      {result.warnings?.length > 0 && (
        <ul className="warnings" aria-label="Things to be careful about">
          {result.warnings.map((w, i) => <li key={i}><span aria-hidden="true">!</span>{w}</li>)}
        </ul>
      )}

      <div className="kn-grid">
        {result.key_numbers.map((k, i) => (
          <div className="kn" key={i}>
            <span className="kn-label">{k.label}</span>
            <span className="kn-value">{k.value}</span>
            {k.note && <span className="kn-note">{k.note}</span>}
          </div>
        ))}
      </div>

      <div className="charts">
        {result.charts.map((c) => <Chart key={c.id} spec={c} />)}
      </div>

      <details className="how">
        <summary>How this was worked out</summary>
        <p>{result.how_it_works}</p>
      </details>
      <button className="link-btn" onClick={() => setShowTech(!showTech)}>{showTech ? "Hide" : "Show"} technical details</button>
      {showTech && <pre className="tech">{JSON.stringify(result.details, null, 2)}</pre>}
    </section>
  );
}
