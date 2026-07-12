import { useEffect, useState } from "react";
import { fetchFairnessAudit } from "../api.js";

const DIMENSION_TITLES = {
  sector_segment: "By sector",
  loan_type_segment: "By loan type",
  data_richness: "By data richness (established vs new-to-credit)",
};

const DISPARITY_THRESHOLD = 1.25;

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);

function DimensionTable({ name, rows }) {
  return (
    <div className="metrics-section">
      <h4>{DIMENSION_TITLES[name] || name}</h4>
      <table>
        <thead>
          <tr>
            <th>Group</th>
            <th>n</th>
            <th>Avg calibrated PD</th>
            <th>High-risk share</th>
            <th>False-positive rate</th>
            <th>Disparity ratio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const flagged = r.disparity_ratio != null && r.disparity_ratio > DISPARITY_THRESHOLD;
            return (
              <tr key={r.group} className={flagged ? "disparity-flag" : ""}>
                <td>{String(r.group).replace(/_/g, " ")}</td>
                <td>{r.n.toLocaleString()}</td>
                <td>{pct(r.avg_calibrated_pd)}</td>
                <td>{pct(r.high_risk_share)}</td>
                <td>{pct(r.false_positive_rate)}</td>
                <td>
                  {r.disparity_ratio == null ? "—" : r.disparity_ratio.toFixed(3)}
                  {flagged && <span className="chip chip-warn">&gt; {DISPARITY_THRESHOLD}</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function FairnessAudit() {
  const [audit, setAudit] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFairnessAudit().then(setAudit).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!audit) return <div className="loading">Computing disparate-impact aggregates…</div>;

  return (
    <div className="main">
      <section className="metrics-card">
        <h3>Fairness &amp; bias audit — disparate impact (plan.md §12.6)</h3>
        <p className="chart-note">
          {audit.basis} · {audit.n_loans.toLocaleString()} scored loans. Disparity ratio compares
          each group&apos;s average calibrated PD against the overall book; rows above{" "}
          {DISPARITY_THRESHOLD} are highlighted for review.
        </p>
        {Object.entries(audit.dimensions || {}).map(([name, rows]) => (
          <DimensionTable key={name} name={name} rows={rows} />
        ))}
        {audit.unavailable_slices?.length > 0 && (
          <div className="unavailable-note">
            <strong>Not yet auditable (honest gap):</strong>{" "}
            {audit.unavailable_slices.join("; ")}. Region and gender-of-promoter slices activate
            once RBI-sandbox data carrying those attributes is connected — the group-by machinery
            above is dimension-agnostic and needs no code change.
          </div>
        )}
        {audit.note && <p className="chart-note">{audit.note}</p>}
      </section>
    </div>
  );
}
