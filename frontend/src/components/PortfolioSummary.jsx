import { GRADE_COLORS } from "./LoanList.jsx";

const GRADES = ["A", "B", "C", "D", "E", "F", "G"];

function DriftChip({ drift }) {
  if (!drift) return null;
  if (drift.status === "not_yet_generated") {
    return <span className="chip chip-muted">Drift: not yet generated</span>;
  }
  const drifted = drift.drifted_features?.length || 0;
  const calBad = !["ok", "stable"].includes(drift.calibration_status);
  const bad = drifted > 0 || drift.retrain_triggered || calBad;
  return (
    <span className={`chip ${bad ? "chip-warn" : "chip-ok"}`} title={drift.plain_language}>
      Drift: {drifted > 0 ? `${drifted} feature${drifted > 1 ? "s" : ""} drifted` : "stable"}
      {drift.retrain_triggered ? " · retrain triggered" : ""}
    </span>
  );
}

export default function PortfolioSummary({ summary, drift }) {
  if (!summary) return null;
  const shares = summary.grade_shares || {};
  const maxShare = Math.max(...GRADES.map((g) => shares[g] || 0), 0.01);
  return (
    <div className="portfolio-summary">
      <h2>Portfolio summary</h2>
      <div className="grade-bars">
        {GRADES.map((g) => (
          <div className="grade-bar-row" key={g}>
            <span className="grade-bar-label" style={{ color: GRADE_COLORS[g] }}>
              {g}
            </span>
            <div className="grade-bar-track">
              <div
                className="grade-bar-fill"
                style={{
                  width: `${((shares[g] || 0) / maxShare) * 100}%`,
                  background: GRADE_COLORS[g],
                }}
              />
            </div>
            <span className="grade-bar-value">
              {((shares[g] || 0) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
      <div className="el-line">
        Expected loss rate: <strong>{(summary.expected_loss_rate * 100).toFixed(2)}%</strong>
        <span className="el-note" title={summary.el_convention?.formula}>
          {summary.el_convention?.note}
        </span>
      </div>
      <div className="summary-meta">
        {summary.n_loans.toLocaleString()} scored loans (OOT test slice)
      </div>
      <DriftChip drift={drift} />
    </div>
  );
}
