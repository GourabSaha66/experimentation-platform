// Thin wrapper around the FastAPI service. Errors carry a plain-English message.
async function handle(res) {
  if (res.ok) return res.json();
  let msg = `The server returned an error (${res.status}).`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") msg = body.detail;
    else if (Array.isArray(body.detail)) msg = body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  } catch (_) { /* keep default */ }
  throw new Error(msg);
}

export const getCatalog = () => fetch("/api/methods").then(handle);

export const postPower = (body) =>
  fetch("/api/power", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle);

export function inspectFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  return fetch("/api/inspect", { method: "POST", body: fd }).then(handle);
}

export function analyze(method, file, params) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("params", JSON.stringify(params));
  return fetch(`/api/analyze/${method}`, { method: "POST", body: fd }).then(handle);
}

export async function fetchSample(name) {
  const res = await fetch(`/api/samples/${name}`);
  if (!res.ok) throw new Error("Could not load that example.");
  const blob = await res.blob();
  return new File([blob], name, { type: "text/csv" });
}
