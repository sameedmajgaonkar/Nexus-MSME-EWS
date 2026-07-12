import { useState } from "react";
import { scoreLoan } from "../api.js";
import { GRADE_COLORS } from "./LoanList.jsx";

// Plain language only (plan.md §18.2): no PD decimals, no SHAP, no model internals.
const BAND_LANGUAGE = {
  A: "routine monitoring — no action needed on your side",
  B: "routine monitoring — no action needed on your side",
  C: "monthly review — your lender will check in with you more often",
  D: "monthly review — your lender will check in with you more often",
  E: "enhanced monitoring — your lender may reach out to discuss support options",
  F: "enhanced monitoring — your lender may reach out to discuss support options",
  G: "immediate attention — please contact your relationship manager",
};

function BorrowerCounterfactual({ cf }) {
  if (!cf || !cf.feature) {
    return (
      <p>
        A single specific improvement suggestion is not available for your account type right
        now — your relationship manager can walk you through what would help.
      </p>
    );
  }
  return (
    <p>
      Improving <strong>{cf.label.toLowerCase()}</strong> from {cf.current_value} to{" "}
      {cf.suggested_value} would move your account to band <strong>{cf.new_grade}</strong>.
    </p>
  );
}

export default function BorrowerView({ loans }) {
  const [loanId, setLoanId] = useState("");
  const [score, setScore] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function check() {
    if (!loanId) return;
    setBusy(true);
    setError(null);
    setScore(null);
    try {
      setScore(await scoreLoan(loanId));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="main borrower-view">
      <div className="disclosure borrower-disclosure">
        ⚠ This assessment used AI. A human officer holds final decision authority — you may
        request a human review of any decision about your account at any time.
      </div>

      <section className="score-card">
        <h2>Your account status</h2>
        <div className="simulate-row">
          <select value={loanId} onChange={(e) => setLoanId(e.target.value)}>
            <option value="">Select your account…</option>
            {loans.map((l) => (
              <option key={l.loan_id} value={l.loan_id}>
                Account #{l.loan_id}
              </option>
            ))}
          </select>
          <input
            className="delta-input"
            placeholder="…or enter your account number"
            value={loanId}
            onChange={(e) => setLoanId(e.target.value.replace(/\D/g, ""))}
          />
          <button className="btn accept" disabled={busy || !loanId} onClick={check}>
            {busy ? "Checking…" : "Check status"}
          </button>
        </div>
        {busy && (
          <p className="chart-note">This can take up to half a minute for newer accounts.</p>
        )}
        {error && <div className="error">{error}</div>}
      </section>

      {score && (
        <section className="score-card">
          <div className="score-header">
            <div
              className="grade-circle"
              style={{ background: GRADE_COLORS[score.risk_grade] }}
            >
              {score.risk_grade}
            </div>
            <div>
              <h2>
                Your account is in band {score.risk_grade} —{" "}
                {BAND_LANGUAGE[score.risk_grade]?.split(" — ")[0]}
              </h2>
              <div className="segment-line">{BAND_LANGUAGE[score.risk_grade]}</div>
            </div>
          </div>
          <div className="detail-block">
            <h4>What would improve this</h4>
            <BorrowerCounterfactual cf={score.counterfactual} />
          </div>
        </section>
      )}
    </div>
  );
}
