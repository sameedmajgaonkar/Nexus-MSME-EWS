import { useState } from "react";

export default function OverridePanel({ onDecision, overrides }) {
  const [mode, setMode] = useState(null); // null | "modify" | "override"
  const [reason, setReason] = useState("");
  const [delta, setDelta] = useState("");
  const [status, setStatus] = useState(null);

  async function decide(decision, reasonText, deltaText) {
    try {
      await onDecision(decision, reasonText, deltaText);
      setStatus(
        decision === "accept"
          ? "Decision logged: accepted."
          : `Decision logged: ${decision} with reason (audit trail).`
      );
      setMode(null);
      setReason("");
      setDelta("");
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
  }

  return (
    <section className="override-card">
      <h3>Officer decision</h3>
      <div className="override-buttons">
        <button className="btn accept" onClick={() => decide("accept")}>
          Accept AI assessment
        </button>
        <button
          className={`btn ${mode === "modify" ? "active" : ""}`}
          onClick={() => setMode("modify")}
        >
          Modify…
        </button>
        <button
          className={`btn override ${mode === "override" ? "active" : ""}`}
          onClick={() => setMode("override")}
        >
          Override…
        </button>
      </div>

      {mode && (
        <div className="reason-box">
          {mode === "modify" && (
            <input
              className="delta-input"
              placeholder="Modification delta — what changed (e.g. grade C → D, limit reduced 20%)"
              value={delta}
              onChange={(e) => setDelta(e.target.value)}
            />
          )}
          <textarea
            placeholder={`Reason is mandatory for a ${mode} (logged to the append-only audit trail)`}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={3}
          />
          <div className="override-buttons">
            <button
              className={`btn ${mode === "override" ? "override" : "accept"}`}
              disabled={!reason.trim()}
              onClick={() => decide(mode, reason, mode === "modify" ? delta : undefined)}
            >
              Confirm {mode}
            </button>
            <button className="btn" onClick={() => setMode(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {status && <div className="status-line">{status}</div>}

      {overrides.length > 0 && (
        <details className="audit-log">
          <summary>Audit log ({overrides.length} recent decisions)</summary>
          <ul>
            {overrides.map((o) => (
              <li key={o.id}>
                <code>{o.created_at}</code> — loan #{o.loan_id}: <strong>{o.decision}</strong>
                {o.reason ? ` — "${o.reason}"` : ""}
                {o.delta ? ` [delta: ${o.delta}]` : ""} (grade {o.risk_grade ?? "?"})
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
