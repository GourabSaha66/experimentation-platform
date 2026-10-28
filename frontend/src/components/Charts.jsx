import { useMemo, useState } from "react";

/* Small, dependency-free SVG charts. Every chart is described by a plain JSON spec from the API:
   line (with optional bands, dashed/muted series, reference lines), bar (grouped, stacked, horizontal)
   and forest (estimate with interval). Colours come from CSS variables so light and dark both work. */

const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"];
const W = 640;

export function fmtTick(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
  if (a >= 1000) return Math.round(v).toLocaleString();
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1).replace(/\.0$/, "");
  if (a >= 1) return v.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  if (a >= 0.001) return v.toPrecision(2).replace(/0+$/, "").replace(/\.$/, "");
  return v.toExponential(0);
}

// x axes often hold years or period numbers, which should not get thousands separators
export function fmtX(v) {
  return Number.isInteger(v) && Math.abs(v) < 100000 ? String(v) : fmtTick(v);
}

function niceTicks(min, max, count = 5) {
  if (!(max > min)) { max = min + 1; }
  const span = max - min;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const out = [];
  for (let v = start; v <= max + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <span key={i} className="legend-item">
          <i style={{ background: it.color, borderTop: it.dashed ? "2px dashed " + it.color : undefined, height: it.dashed ? 0 : 10 }} />
          {it.name}
        </span>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- line */
function LineChart({ spec }) {
  const H = 320;
  const m = { l: 62, r: 18, t: 14, b: 48 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const [hover, setHover] = useState(null);
  const series = spec.series || [];
  const logY = !!spec.log_y;

  const geo = useMemo(() => {
    let xs = [], ys = [];
    series.forEach((s) => s.points.forEach((p, k) => {
      xs.push(p.x);
      // wide early bands (few observations) would flatten everything: size the axis from the later half
      const bandOk = k >= s.points.length / 2;
      [p.y, bandOk ? p.lo : null, bandOk ? p.hi : null].forEach((v) => { if (v !== null && v !== undefined && (!logY || v > 0)) ys.push(v); });
    }));
    (spec.hlines || []).forEach((h) => { if (!logY || h.y > 0) ys.push(h.y); });
    (spec.vlines || []).forEach((v) => xs.push(v.x));
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (logY) {
      y0 = Math.pow(10, Math.floor(Math.log10(y0))); y1 = Math.pow(10, Math.ceil(Math.log10(y1)));
      if (y0 === y1) y1 = y0 * 10;
    } else {
      const pad = (y1 - y0) * 0.06 || 1;
      if (!spec.series.some((s) => s.fill)) { y0 -= pad; y1 += pad; } else { y0 = Math.min(0, y0); y1 += pad; }
    }
    return { x0, x1, y0, y1 };
  }, [spec]);

  const sx = (x) => m.l + ((x - geo.x0) / (geo.x1 - geo.x0)) * iw;
  const clampY = (v) => Math.min(m.t + ih, Math.max(m.t, v));
  const sy = (y) => {
    if (logY) { const v = Math.max(y, geo.y0); return m.t + ih - ((Math.log10(v) - Math.log10(geo.y0)) / (Math.log10(geo.y1) - Math.log10(geo.y0))) * ih; }
    return m.t + ih - ((y - geo.y0) / (geo.y1 - geo.y0)) * ih;
  };
  const xt = niceTicks(geo.x0, geo.x1, 6);
  let yt;
  if (logY) { yt = []; for (let e = Math.log10(geo.y0); e <= Math.log10(geo.y1) + 1e-9; e++) yt.push(Math.pow(10, e)); }
  else yt = niceTicks(geo.y0, geo.y1, 5);

  const visible = series.filter((s) => !s.muted && s.name);
  const hoverXs = useMemo(() => {
    const main = series.find((s) => !s.muted) || series[0];
    return main ? main.points.map((p) => p.x) : [];
  }, [spec]);

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const xv = geo.x0 + ((px - m.l) / iw) * (geo.x1 - geo.x0);
    let best = null, bd = Infinity;
    hoverXs.forEach((x) => { const d = Math.abs(x - xv); if (d < bd) { bd = d; best = x; } });
    setHover(best);
  }

  const hoverRows = hover === null ? [] : series.filter((s) => !s.muted && s.name).map((s) => {
    const p = s.points.find((q) => q.x === hover);
    return p ? { name: s.name, y: p.y, lo: p.lo, hi: p.hi } : null;
  }).filter(Boolean);

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="svg" role="img" aria-label={spec.title}>
        {yt.map((v, i) => (
          <g key={"y" + i}>
            <line x1={m.l} x2={W - m.r} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={m.l - 8} y={sy(v) + 4} textAnchor="end" className="tick">{fmtTick(v)}</text>
          </g>
        ))}
        {xt.map((v, i) => (
          <text key={"x" + i} x={sx(v)} y={H - m.b + 18} textAnchor="middle" className="tick">{fmtX(v)}</text>
        ))}
        <line x1={m.l} x2={W - m.r} y1={m.t + ih} y2={m.t + ih} className="axis" />
        {spec.x_label && <text x={m.l + iw / 2} y={H - 6} textAnchor="middle" className="axis-label">{spec.x_label}</text>}
        {spec.y_label && <text transform={`translate(14 ${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{spec.y_label}</text>}

        {(spec.hlines || []).map((h, i) => (!logY || h.y > 0) && (
          <g key={"h" + i}>
            <line x1={m.l} x2={W - m.r} y1={sy(h.y)} y2={sy(h.y)} className="refline" />
            {h.label && <text x={W - m.r - 4} y={sy(h.y) - 5} textAnchor="end" className="ref-label">{h.label}</text>}
          </g>
        ))}
        {(spec.vlines || []).map((v, i) => (
          <g key={"v" + i}>
            <line x1={sx(v.x)} x2={sx(v.x)} y1={m.t} y2={m.t + ih} className="refline" />
            {v.label && <text x={sx(v.x) + 5} y={m.t + 12} className="ref-label">{v.label}</text>}
          </g>
        ))}

        {series.map((s, i) => {
          const color = s.muted ? "var(--muted-line)" : COLORS[visibleIndex(series, i) % COLORS.length];
          const pts = s.points.filter((p) => p.y !== null && p.y !== undefined);
          const d = pts.map((p, k) => `${k ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
          const hasBand = pts.some((p) => p.lo !== null && p.lo !== undefined);
          const band = hasBand
            ? pts.filter((p) => p.lo != null && p.hi != null).map((p, k) => `${k ? "L" : "M"}${sx(p.x).toFixed(1)},${clampY(sy(p.hi)).toFixed(1)}`).join(" ") + " " +
              pts.filter((p) => p.lo != null && p.hi != null).reverse().map((p) => `L${sx(p.x).toFixed(1)},${clampY(sy(p.lo)).toFixed(1)}`).join(" ") + " Z"
            : null;
          const area = s.fill ? d + ` L${sx(pts[pts.length - 1].x).toFixed(1)},${sy(Math.max(geo.y0, 0)).toFixed(1)} L${sx(pts[0].x).toFixed(1)},${sy(Math.max(geo.y0, 0)).toFixed(1)} Z` : null;
          return (
            <g key={i}>
              {band && <path d={band} fill={color} opacity="0.16" />}
              {area && <path d={area} fill={color} opacity="0.18" />}
              <path d={d} fill="none" stroke={color} strokeWidth={s.strong ? 2.6 : s.muted ? 1 : 2} strokeDasharray={s.dashed ? "6 4" : undefined}
                opacity={s.muted ? 0.7 : 1} strokeLinejoin="round" strokeLinecap="round" />
              {pts.length <= 24 && !s.muted && pts.map((p, k) => <circle key={k} cx={sx(p.x)} cy={sy(p.y)} r="2.6" fill={color} />)}
            </g>
          );
        })}

        {hover !== null && <line x1={sx(hover)} x2={sx(hover)} y1={m.t} y2={m.t + ih} className="cursor" />}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </svg>
      {hover !== null && hoverRows.length > 0 && (
        <div className="tooltip" style={{ left: `${Math.min(88, Math.max(12, (sx(hover) / W) * 100))}%` }}>
          <strong>{spec.x_label ? spec.x_label + ": " : ""}{fmtX(hover)}</strong>
          {hoverRows.map((r, i) => (
            <div key={i}>{r.name}: <b>{fmtTick(r.y)}</b>{r.lo != null ? ` (${fmtTick(r.lo)} to ${fmtTick(r.hi)})` : ""}</div>
          ))}
        </div>
      )}
      <Legend items={visible.map((s, i) => ({ name: s.name, color: COLORS[i % COLORS.length], dashed: s.dashed }))} />
    </div>
  );
}

function visibleIndex(series, idx) {
  let n = 0;
  for (let i = 0; i < idx; i++) if (!series[i].muted && series[i].name) n++;
  return n;
}

/* ---------------------------------------------------------------- bar */
function BarChart({ spec }) {
  const cats = spec.categories || [];
  const series = spec.series || [];
  const horizontal = !!spec.horizontal;
  const stacked = !!spec.stacked;
  const H = horizontal ? Math.max(220, 60 + cats.length * (series.length > 1 ? 40 : 28)) : 320;
  const m = horizontal ? { l: 130, r: 20, t: 12, b: 40 } : { l: 62, r: 18, t: 14, b: cats.length > 8 ? 78 : 56 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const totals = cats.map((_, i) => series.reduce((a, s) => a + (s.values[i] || 0), 0));
  let max = stacked ? Math.max(...totals) : Math.max(...series.flatMap((s) => s.values), spec.hline || 0);
  max = max * 1.06 || 1;
  const ticks = niceTicks(0, max, 5);
  const top = ticks[ticks.length - 1] > max ? ticks[ticks.length - 1] : max;
  const bandN = cats.length;
  const [tip, setTip] = useState(null);

  const val = (v) => (horizontal ? (v / top) * iw : (v / top) * ih);
  const label = (c) => (String(c).length > 16 && !horizontal ? String(c).slice(0, 15) + "…" : String(c));

  let markerIdx = null;
  if (spec.marker) {
    let bd = Infinity;
    cats.forEach((c, i) => { const d = Math.abs(parseFloat(c) - spec.marker.x); if (d < bd) { bd = d; markerIdx = i; } });
  }

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="svg" role="img" aria-label={spec.title}>
        {ticks.map((v, i) => horizontal ? (
          <g key={i}>
            <line x1={m.l + val(v)} x2={m.l + val(v)} y1={m.t} y2={m.t + ih} className="grid" />
            <text x={m.l + val(v)} y={m.t + ih + 16} textAnchor="middle" className="tick">{fmtTick(v)}</text>
          </g>
        ) : (
          <g key={i}>
            <line x1={m.l} x2={W - m.r} y1={m.t + ih - val(v)} y2={m.t + ih - val(v)} className="grid" />
            <text x={m.l - 8} y={m.t + ih - val(v) + 4} textAnchor="end" className="tick">{fmtTick(v)}</text>
          </g>
        ))}
        {cats.map((c, i) => {
          const band = (horizontal ? ih : iw) / bandN;
          const inner = Math.min(band * 0.78, 64);
          const bw = stacked ? inner : inner / series.length;
          let acc = 0;
          return (
            <g key={i} onMouseEnter={() => setTip(i)} onMouseLeave={() => setTip(null)}>
              {series.map((s, k) => {
                const v = s.values[i] || 0;
                const color = COLORS[k % COLORS.length];
                const off = (band - inner) / 2 + (stacked ? 0 : k * bw);
                let rect;
                if (horizontal) rect = <rect x={m.l} y={m.t + i * band + off} width={Math.max(val(v), 0)} height={bw - 2} fill={color} rx="2" />;
                else if (stacked) { rect = <rect x={m.l + i * band + off} y={m.t + ih - val(acc + v)} width={bw} height={Math.max(val(v), 0)} fill={color} />; acc += v; }
                else rect = <rect x={m.l + i * band + off} y={m.t + ih - val(v)} width={bw - 2} height={Math.max(val(v), 0)} fill={color} rx="2" />;
                return <g key={k}>{rect}</g>;
              })}
              {horizontal
                ? <text x={m.l - 8} y={m.t + i * band + band / 2 + 4} textAnchor="end" className="tick">{label(c)}</text>
                : <text x={m.l + i * band + band / 2} y={m.t + ih + 16} textAnchor={cats.length > 8 ? "end" : "middle"} className="tick"
                    transform={cats.length > 8 ? `rotate(-40 ${m.l + i * band + band / 2} ${m.t + ih + 16})` : undefined}>{label(c)}</text>}
              <rect x={horizontal ? m.l : m.l + i * band} y={horizontal ? m.t + i * band : m.t} width={horizontal ? iw : band} height={horizontal ? band : ih} fill="transparent" />
            </g>
          );
        })}
        {spec.hline != null && (horizontal
          ? <line x1={m.l + val(spec.hline)} x2={m.l + val(spec.hline)} y1={m.t} y2={m.t + ih} className="refline" />
          : <line x1={m.l} x2={W - m.r} y1={m.t + ih - val(spec.hline)} y2={m.t + ih - val(spec.hline)} className="refline" />)}
        {markerIdx !== null && !horizontal && (
          <g>
            <line x1={m.l + (markerIdx + 0.5) * (iw / bandN)} x2={m.l + (markerIdx + 0.5) * (iw / bandN)} y1={m.t} y2={m.t + ih} className="marker" />
            <text x={m.l + (markerIdx + 0.5) * (iw / bandN) + 5} y={m.t + 12} className="ref-label">{spec.marker.label}</text>
          </g>
        )}
        <line x1={m.l} x2={W - m.r} y1={m.t + ih} y2={m.t + ih} className="axis" />
        {spec.y_label && (horizontal
          ? <text x={m.l + iw / 2} y={H - 4} textAnchor="middle" className="axis-label">{spec.y_label}</text>
          : <text transform={`translate(14 ${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{spec.y_label}</text>)}
      </svg>
      {tip !== null && (
        <div className="tooltip" style={{ left: horizontal ? "50%" : `${Math.min(88, Math.max(12, ((m.l + (tip + 0.5) * (iw / bandN)) / W) * 100))}%` }}>
          <strong>{cats[tip]}</strong>
          {series.map((s, k) => <div key={k}>{s.name}: <b>{fmtTick(s.values[tip])}</b></div>)}
        </div>
      )}
      {(series.length > 1 || spec.hline != null) && (
        <Legend items={[...series.map((s, i) => ({ name: s.name, color: COLORS[i % COLORS.length] })),
          ...(spec.hline != null ? [{ name: `reference ${fmtTick(spec.hline)}`, color: "var(--ref)", dashed: true }] : [])]} />
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- forest */
function ForestChart({ spec }) {
  const rows = spec.rows || [];
  const rowH = 44;
  const H = 40 + rows.length * rowH + 34;
  const m = { l: 250, r: 28, t: 12, b: 34 };
  const iw = W - m.l - m.r;
  const vals = rows.flatMap((r) => [r.estimate, r.lo, r.hi]).filter((v) => v !== null && v !== undefined);
  if (spec.ref !== null && spec.ref !== undefined) vals.push(spec.ref);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const sx = (v) => m.l + ((v - lo) / (hi - lo)) * iw;
  const ticks = niceTicks(lo, hi, 5);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="svg" role="img" aria-label={spec.title}>
        {ticks.map((v, i) => (
          <g key={i}>
            <line x1={sx(v)} x2={sx(v)} y1={m.t} y2={H - m.b} className="grid" />
            <text x={sx(v)} y={H - m.b + 16} textAnchor="middle" className="tick">{fmtTick(v)}</text>
          </g>
        ))}
        {spec.ref !== null && spec.ref !== undefined && (
          <g>
            <line x1={sx(spec.ref)} x2={sx(spec.ref)} y1={m.t} y2={H - m.b} className="refline" />
            {spec.ref_label && <text x={sx(spec.ref) + 5} y={m.t + 10} className="ref-label">{spec.ref_label}</text>}
          </g>
        )}
        {rows.map((r, i) => {
          const y = m.t + 24 + i * rowH;
          const color = COLORS[0];
          return (
            <g key={i}>
              <text x={m.l - 12} y={y + 4} textAnchor="end" className="row-label">{r.label}</text>
              {r.lo !== null && r.lo !== undefined && r.hi !== null && r.hi !== undefined && (
                <g>
                  <line x1={sx(r.lo)} x2={sx(r.hi)} y1={y} y2={y} stroke={color} strokeWidth="3" strokeLinecap="round" />
                  <line x1={sx(r.lo)} x2={sx(r.lo)} y1={y - 6} y2={y + 6} stroke={color} strokeWidth="2" />
                  <line x1={sx(r.hi)} x2={sx(r.hi)} y1={y - 6} y2={y + 6} stroke={color} strokeWidth="2" />
                </g>
              )}
              <circle cx={sx(r.estimate)} cy={y} r="6" fill={r.lo == null ? "var(--c2)" : color} stroke="var(--surface)" strokeWidth="2" />
              <text x={sx(r.estimate)} y={y - 12} textAnchor="middle" className="val-label">{fmtTick(r.estimate)}</text>
            </g>
          );
        })}
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} className="axis" />
      </svg>
      {rows.some((r) => r.lo == null) && <p className="chart-note">Dots without a bar have no interval from that method.</p>}
    </div>
  );
}

export default function Chart({ spec }) {
  if (!spec) return null;
  return (
    <figure className="chart-card">
      <figcaption>{spec.title}</figcaption>
      {spec.kind === "line" && <LineChart spec={spec} />}
      {spec.kind === "bar" && <BarChart spec={spec} />}
      {spec.kind === "forest" && <ForestChart spec={spec} />}
    </figure>
  );
}
