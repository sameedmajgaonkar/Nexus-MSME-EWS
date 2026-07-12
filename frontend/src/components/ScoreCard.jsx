import { GRADE_COLORS } from "./LoanList.jsx";

const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;

function Counterfactual({ cf, grade }) {
  if (!cf) return null;
  if (!cf.feature) {
    return (
      <div className="detail-block">
        <h4>Counterfactual</h4>
        <p className="cf-unavailable">{cf.reason}</p>
      </div>
    );
  }
  return (
    <div className="detail-block">
      <h4>Counterfactual</h4>
      <p>
        If <strong>{cf.label}</strong> moved from {cf.current_value} to {cf.suggested_value},
        12-month PD falls to <strong>{pct(cf.new_pd)}</strong> (grade {grade} →{" "}
        <strong>{cf.new_grade}</strong>).
      </p>
    </div>
  );
}

export default function ScoreCard({ score }) {
  const band = score.confidence_band;
  const narrative = score.narrative;
  return (
    <section className="score-card">
      <div className="score-header">
        <div
          className="grade-circle"
          style={{ background: GRADE_COLORS[score.risk_grade] }}
        >
          {score.risk_grade}
        </div>
        <div>
          <h2>Loan #{score.loan_id}</h2>
          <div className="segment-line">
            {score.segment === "working_capital_proxy" ? "Working capital" : "Term loan"} ·{" "}
            {score.sector?.replace(/_/g, " ")} ·{" "}
            {score.data_richness === "established" ? "Established" : "New-to-credit"}
          </div>
          <div className="chip-row">
            <span className="chip chip-model">
              {score.model_used === "tabpfn_thin_file"
                ? "TabPFN thin-file specialist"
                : "Discrete-time hazard (LightGBM)"}
            </span>
            {band?.wide_band_flag && (
              <span className="chip chip-review">Mandatory human review — wide band</span>
            )}
          </div>
        </div>
        <div className="pd-block">
          <div className="pd-value">
            {pct(score.calibrated_pd_12m)}
            {band && <span className="pd-band"> ± {pct(band.half_width)}</span>}
          </div>
          <div className="pd-label">
            Calibrated 12-month PD
            {band && ` (n=${band.n_comparable} comparable)`}
          </div>
          <div className="pd-raw">raw score {pct(score.raw_cum_pd_12m)}</div>
        </div>
      </div>

      {narrative && (
        <div className="detail-block">
          <h4>
            Narrative
            <span
              className={`chip ${narrative.verified ? "chip-ok" : "chip-warn"}`}
              title="Every narrative is checked against the SHAP attributions before display"
            >
              {narrative.source === "template" ? "template-verified" : "LLM"}
              {narrative.verified ? " ✓" : " — unverified"}
            </span>
          </h4>
          <p className="narrative-text">{narrative.text}</p>
        </div>
      )}

      <Counterfactual cf={score.counterfactual} grade={score.risk_grade} />

      <div className="action-line">
        <strong>Recommended action:</strong> {score.recommended_action}
      </div>
      <div className="disclosure-inline">{score.disclosure}</div>
    </section>
  );
}
