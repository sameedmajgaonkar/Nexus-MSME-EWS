const METRIC_COLS = ["auc_roc", "ks", "pr_auc", "recall_at_fpr10", "brier"];
const SECTION_TITLES = {
  phase2_metrics: "Baseline scorecard — outcome: current-loan default (OOT test)",
  phase3_metrics_target: "Baseline vs hazard — outcome: current-loan default (OOT test)",
  phase3_metrics_delinquency: "Hazard model — outcome: serious delinquency within 12m (OOT test)",
  phase4_metrics: "Calibration effect — outcome: serious delinquency within 12m (OOT test)",
};

function Section({ name, table }) {
  const models = Object.keys(table[METRIC_COLS[0]] || {});
  return (
    <div className="metrics-section">
      <h4>{SECTION_TITLES[name] || name}</h4>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {METRIC_COLS.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m} className={m.startsWith("naive") ? "naive-row" : ""}>
              <td>{m}</td>
              {METRIC_COLS.map((c) => (
                <td key={c}>{table[c]?.[m] ?? "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function MetricsTable({ metrics }) {
  return (
    <section className="metrics-card">
      <h3>Model comparison — always shown against the naive baseline</h3>
      <p className="chart-note">
        Accuracy is deliberately absent: a model that predicts "no default" for every loan scores
        ~92% accuracy and catches nothing. These are the metrics that matter for rare events.
      </p>
      {Object.entries(metrics).map(([name, table]) => (
        <Section key={name} name={name} table={table} />
      ))}
    </section>
  );
}
