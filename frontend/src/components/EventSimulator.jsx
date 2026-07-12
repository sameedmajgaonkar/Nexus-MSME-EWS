import { useState } from "react";
import { simulateEvent } from "../api.js";

const EVENT_OPTIONS = [
  { value: "gst_filing", label: "GST filing — reported turnover drops ~30%" },
  { value: "bank_transaction", label: "Bank transaction — overdue signal appears" },
  { value: "repayment", label: "Repayment — on-time payment improves history" },
];

export default function EventSimulator({ loanId, onPublished }) {
  const [type, setType] = useState("gst_filing");
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  async function publish() {
    setBusy(true);
    setStatus(null);
    try {
      const res = await simulateEvent({ loan_id: loanId, type });
      setStatus(
        `Event #${res.event_id} published to the bus — the consumer re-scores loan #${loanId}; ` +
          "a grade move raises an alert within ~5s, otherwise the trajectory updates silently."
      );
      onPublished?.();
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="override-card">
      <h3>Simulate a live event</h3>
      <p className="chart-note">
        Publishes a simulated event onto the streaming bus — no manual re-scoring step.
      </p>
      <div className="simulate-row">
        <select value={type} onChange={(e) => setType(e.target.value)}>
          {EVENT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <button className="btn" disabled={busy} onClick={publish}>
          {busy ? "Publishing…" : "Simulate event"}
        </button>
      </div>
      {status && <div className="status-line">{status}</div>}
    </section>
  );
}
