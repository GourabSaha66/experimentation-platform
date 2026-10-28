import { SPECS } from "../specs.js";

function pickValue(f, vals, taken) {
  const rest = vals.filter((v) => !taken.includes(v));
  const pref = f.prefer ? rest.find((v) => f.prefer.test(v)) : null;
  return pref || (f.key === "treatment" || f.key === "treated_value" ? rest[rest.length - 1] : rest[0]);
}

/* Pick sensible defaults from the column names and contents, so most files need little or no clicking. */
export function guessParams(method, columns) {
  const spec = SPECS[method];
  const params = {};
  const used = new Set();
  const byName = Object.fromEntries(columns.map((c) => [c.name, c]));
  for (const f of spec.fields) {
    if (f.default !== undefined) params[f.key] = f.default;
    if (f.kind === "column" || f.kind === "multicolumn") {
      if (f.optional && !f.hint) continue;
      let cands = columns.filter((c) => (f.filter ? f.filter(c) : true));
      const free = cands.filter((c) => !used.has(c.name));
      const named = free.filter((c) => f.hint && f.hint.test(c.name));
      if (f.kind === "multicolumn") {
        const pick = (named.length ? named : free).map((c) => c.name).slice(0, 12);
        if (pick.length) params[f.key] = pick;
        pick.forEach((n) => used.add(n));
      } else {
        const pick = named[0] || (f.optional ? null : free[0]);
        if (pick) { params[f.key] = pick.name; used.add(pick.name); }
      }
    }
  }
  // values depend on the columns picked above
  for (const f of spec.fields) {
    if (f.kind !== "value") continue;
    const col = byName[params[f.of]];
    const vals = col ? col.values : [];
    if (!vals.length || f.optional) continue;
    const taken = spec.fields.filter((g) => g.kind === "value" && g.of === f.of && g.key !== f.key && params[g.key]).map((g) => params[g.key]);
    params[f.key] = pickValue(f, vals, taken);
  }
  return params;
}

function Field({ f, params, set, columns }) {
  const value = params[f.key] ?? "";
  const label = (
    <span className="lbl">{f.label}{f.optional && <em> optional</em>}</span>
  );
  const help = f.help ? <small>{f.help}</small> : null;
  if (f.kind === "column") {
    const opts = columns.filter((c) => (f.filter ? f.filter(c) : true));
    // keep the current choice visible even if the filter would hide it
    const names = new Set(opts.map((o) => o.name));
    if (value && !names.has(value)) { const c = columns.find((x) => x.name === value); if (c) opts.push(c); }
    return (
      <label className="field">{label}
        <select value={value} onChange={(e) => set(f.key, e.target.value)}>
          <option value="">{f.optional ? "None" : "Choose a column"}</option>
          {opts.map((c) => <option key={c.name} value={c.name}>{c.name}  ({c.dtype === "number" ? "number" : c.n_unique + " values"})</option>)}
        </select>{help}
      </label>
    );
  }
  if (f.kind === "multicolumn") {
    // a column already used as group/treatment/outcome cannot also be a metric or covariate
    const taken = ["group_col", "treatment", "outcome", "time_col", "unit_col"].map((k) => params[k]).filter(Boolean);
    const opts = columns.filter((c) => (f.filter ? f.filter(c) : true) && !taken.includes(c.name));
    const sel = params[f.key] || [];
    const toggle = (n) => set(f.key, sel.includes(n) ? sel.filter((x) => x !== n) : [...sel, n]);
    return (
      <div className="field">{label}{help}
        <div className="checks">
          {opts.map((c) => (
            <label key={c.name} className={"check" + (sel.includes(c.name) ? " on" : "")}>
              <input type="checkbox" checked={sel.includes(c.name)} onChange={() => toggle(c.name)} />{c.name}
            </label>
          ))}
        </div>
      </div>
    );
  }
  if (f.kind === "value") {
    const col = columns.find((c) => c.name === params[f.of]);
    const vals = col ? col.values : [];
    if (vals.length) {
      return (
        <label className="field">{label}
          <select value={value} onChange={(e) => set(f.key, e.target.value)}>
            <option value="">{f.optional ? "Detect automatically" : "Choose a value"}</option>
            {vals.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>{help}
        </label>
      );
    }
    return (
      <label className="field">{label}
        <input type="text" value={value} onChange={(e) => set(f.key, e.target.value)} placeholder="Type the value exactly" />{help}
      </label>
    );
  }
  if (f.kind === "select") {
    return (
      <label className="field">{label}
        <select value={String(value)} onChange={(e) => set(f.key, f.options.find((o) => String(o.v) === e.target.value).v)}>
          {f.options.map((o) => <option key={String(o.v)} value={String(o.v)}>{o.t}</option>)}
        </select>{help}
      </label>
    );
  }
  if (f.kind === "number") {
    return (
      <label className="field">{label}
        <input type="number" step="any" value={value} onChange={(e) => set(f.key, e.target.value === "" ? "" : Number(e.target.value))} />{help}
      </label>
    );
  }
  if (f.kind === "bool") {
    return (
      <label className="field inline"><input type="checkbox" checked={!!params[f.key]} onChange={(e) => set(f.key, e.target.checked)} />{label}{help}</label>
    );
  }
  return null;
}

export default function SettingsForm({ method, columns, params, setParams }) {
  const spec = SPECS[method];
  const set = (k, v) => setParams((p) => {
    const next = { ...p, [k]: v };
    // changing a column re-picks the group values that depend on it
    const dependants = spec.fields.filter((f) => f.kind === "value" && f.of === k && !f.optional);
    const col = columns.find((c) => c.name === v);
    spec.fields.filter((f) => f.kind === "value" && f.of === k).forEach((f) => { next[f.key] = ""; });
    dependants.forEach((f) => {
      const vals = col ? col.values : [];
      if (!vals.length) return;
      const taken = dependants.filter((h) => h.key !== f.key && next[h.key]).map((h) => next[h.key]);
      next[f.key] = pickValue(f, vals, taken);
    });
    return next;
  });
  return (
    <div className="form-grid">
      {spec.fields.filter((f) => !f.showIf || f.showIf(params)).map((f) => (
        <Field key={f.key} f={f} params={params} set={set} columns={columns} />
      ))}
    </div>
  );
}

/* Turn the form state into the request body: drop empty optional values, keep numbers as numbers. */
export function cleanParams(method, params) {
  const spec = SPECS[method];
  const out = {};
  for (const f of spec.fields) {
    if (f.showIf && !f.showIf(params)) continue;
    const v = params[f.key];
    if (v === "" || v === undefined || v === null) continue;
    if (Array.isArray(v) && v.length === 0) continue;
    out[f.key] = v;
  }
  return out;
}

export function missingRequired(method, params) {
  const spec = SPECS[method];
  return spec.fields.filter((f) => !f.optional && (!f.showIf || f.showIf(params)) && f.kind !== "bool" && f.kind !== "select" &&
    (params[f.key] === undefined || params[f.key] === "" || (Array.isArray(params[f.key]) && params[f.key].length === 0))).map((f) => f.label);
}
