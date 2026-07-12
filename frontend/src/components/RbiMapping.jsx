import { useEffect, useState } from "react";
import { fetchRbiMapping } from "../api.js";

const STATE_COLORS = {
  Performing: "#2e9e5b",
  WatchList: "#d8b722",
  HighAlert: "#e08c1f",
  Critical: "#d43a2f",
  NPA_Stage3: "#a01f5c",
};

function StateFlow({ machine }) {
  if (!machine) return null;
  const forward = machine.transitions.filter((t) =>
    machine.states.indexOf(t.to) > machine.states.indexOf(t.from)
  );
  const backward = machine.transitions.filter(
    (t) => machine.states.indexOf(t.to) < machine.states.indexOf(t.from)
  );
  return (
    <div className="metrics-section">
      <h4>Early-warning state machine (forward-looking ML layer → regulatory states)</h4>
      <div className="state-flow">
        {machine.states.map((s, i) => (
          <span key={s} className="state-flow-step">
            <span className="state-box" style={{ borderColor: STATE_COLORS[s] || "#4a5670" }}>
              {s.replace(/_/g, " ")}
            </span>
            {i < machine.states.length - 1 && <span className="state-arrow">→</span>}
          </span>
        ))}
      </div>
      <table>
        <thead>
          <tr>
            <th>From</th>
            <th>To</th>
            <th>Condition</th>
          </tr>
        </thead>
        <tbody>
          {[...forward, ...backward].map((t) => (
            <tr key={`${t.from}-${t.to}`} className={backward.includes(t) ? "naive-row" : ""}>
              <td>{t.from.replace(/_/g, " ")}</td>
              <td>{t.to.replace(/_/g, " ")}</td>
              <td>{t.condition}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RbiMapping() {
  const [mapping, setMapping] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRbiMapping().then(setMapping).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!mapping) return <div className="loading">Loading RBI mapping…</div>;

  return (
    <div className="main">
      <section className="metrics-card">
        <h3>RBI EWS / SMA / RFA / CRILC mapping — read-only (plan.md §12.4)</h3>
        <p className="chart-note">{mapping.positioning}</p>

        <div className="metrics-section">
          <h4>ML-derived trigger → RBI indicator</h4>
          <table>
            <thead>
              <tr>
                <th>ML-derived trigger</th>
                <th>Mapped RBI indicator</th>
                <th>Regulatory action this enables</th>
              </tr>
            </thead>
            <tbody>
              {mapping.ml_trigger_mapping.map((m) => (
                <tr key={m.ml_trigger}>
                  <td>{m.ml_trigger}</td>
                  <td>{m.rbi_indicator}</td>
                  <td>{m.regulatory_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="metrics-section">
          <h4>RBI SMA ladder (the existing, reactive regulatory EWS)</h4>
          <table>
            <thead>
              <tr>
                <th>Bucket</th>
                <th>Definition</th>
              </tr>
            </thead>
            <tbody>
              {mapping.sma_ladder.map((s) => (
                <tr key={s.bucket}>
                  <td>{s.bucket}</td>
                  <td>{s.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <StateFlow machine={mapping.state_machine} />
        <p className="chart-note">Source: {mapping.source}</p>
      </section>
    </div>
  );
}
