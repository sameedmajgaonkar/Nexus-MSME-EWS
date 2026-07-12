const GRADE_COLORS = {
  A: "#2e9e5b",
  B: "#6cbf3f",
  C: "#d8b722",
  D: "#e08c1f",
  E: "#e05c1f",
  F: "#d43a2f",
  G: "#a01f5c",
};

export default function LoanList({ loans, selectedId, onSelect }) {
  return (
    <div>
      <h2>Portfolio sample</h2>
      <ul className="loan-list">
        {loans.map((l) => (
          <li
            key={l.loan_id}
            className={l.loan_id === selectedId ? "selected" : ""}
            onClick={() => onSelect(l.loan_id)}
          >
            <span className="grade-badge" style={{ background: GRADE_COLORS[l.risk_grade] }}>
              {l.risk_grade}
            </span>
            <span className="loan-id">#{l.loan_id}</span>
            <span className="loan-meta">
              {l.segment === "working_capital_proxy" ? "Working capital" : "Term loan"} ·{" "}
              {(l.calibrated_pd_12m * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export { GRADE_COLORS };
