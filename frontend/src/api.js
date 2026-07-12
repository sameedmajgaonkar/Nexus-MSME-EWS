const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail =
      typeof body.detail === "string" ? body.detail : body.detail && JSON.stringify(body.detail);
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const fetchLoans = () => request("/loans");
export const scoreLoan = (loanId) => request(`/score/${loanId}`, { method: "POST" });
export const fetchMetrics = () => request("/metrics");
export const fetchOverrides = () => request("/overrides");
export const submitDecision = (loanId, decision, reason, delta) =>
  request(`/override/${loanId}`, {
    method: "POST",
    body: JSON.stringify({ decision, reason, delta: delta || null }),
  });

// Phase 9 governance + streaming surfaces
export const fetchPortfolioSummary = () => request("/portfolio/summary");
export const runStressTest = (body) =>
  request("/stress-test", { method: "POST", body: JSON.stringify(body) });
export const fetchFairnessAudit = () => request("/fairness/audit");
export const fetchRbiMapping = () => request("/rbi/mapping");
export const fetchDriftReport = () => request("/drift/report");
export const fetchAlerts = (limit = 20) => request(`/alerts?limit=${limit}`);
export const fetchTimeline = (loanId) => request(`/timeline/${loanId}`);
export const simulateEvent = (body) =>
  request("/events/simulate", { method: "POST", body: JSON.stringify(body) });
