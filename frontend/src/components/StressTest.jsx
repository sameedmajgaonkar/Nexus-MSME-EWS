import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { runStressTest } from "../api.js";
import { GRADE_COLORS } from "./LoanList.jsx";

const GRADES = ["A", "B", "C", "D", "E", "F", "G"];
const SECTORS = [
  "Manufacturing",
  "Retail_Trade",
  "Services_IT",
  "Agriculture_Allied",
  "Other_Public",
];
const SHOCKS = [
  { value: "revenue", label: "Revenue shock (portfolio-wide)" },
  { value: "rate", label: "Interest-rate shock (EMI burden)" },
  { value: "sector_demand", label: "Sector demand shock" },
];

// A +200bps rate move on a mid-tenor loan ≈ +10% EMI burden (see StressIn docstring),
// so slider bps map to the API's fractional annuity-burden increase as bps / 2000.
const bpsToFraction = (bps) => bps / 2000;

function DistributionChart({ title, distribution, max }) {
  const data = GRADES.map((g) => ({ grade: g, count: distribution[g] || 0 }));
  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="grade" />
          <YAxis width={55} domain={[0, max]} allowDecimals={false} />
          <Tooltip formatter={(v) => [v, "loans"]} labelFormatter={(g) => `Grade ${g}`} />
          <Bar dataKey="count">
            {data.map((d) => (
              <Cell key={d.grade} fill={GRADE_COLORS[d.grade]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function StressTest() {
  const [shockType, setShockType] = useState("revenue");
  const [revenuePct, setRevenuePct] = useState(-15); // -50..0 %
  const [rateBps, setRateBps] = useState(200); // 0..+500 bps
  const [sector, setSector] = useState(SECTORS[0]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const isRate = shockType === "rate";
  const magnitude = isRate ? bpsToFraction(rateBps) : revenuePct / 100;

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await runStressTest({
          shock_type: shockType,
          magnitude,
          sector: shockType === "sector_demand" ? sector : undefined,
        })
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const maxCount = result
    ? Math.max(
        ...GRADES.map((g) =>
          Math.max(result.pre_shock_distribution[g] || 0, result.post_shock_distribution[g] || 0)
        )
      )
    : 0;
  const deltaBps = result ? result.delta_expected_loss_rate * 10000 : 0;

  return (
    <div className="main">
      <section className="score-card">
        <h2>Stress-test simulator</h2>
        <p className="chart-note">
          Perturbs the chosen driver across the scored book and re-runs the already-trained
          calibrated model — pure re-inference, no retraining (plan.md §12.7).
        </p>
        <div className="stress-controls">
          <label>
            Shock type
            <select value={shockType} onChange={(e) => setShockType(e.target.value)}>
              {SHOCKS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          {isRate ? (
            <label>
              Magnitude: <strong>+{rateBps} bps</strong> (≈ +{(magnitude * 100).toFixed(1)}% EMI
              burden)
              <input
                type="range"
                min={0}
                max={500}
                step={25}
                value={rateBps}
                onChange={(e) => setRateBps(+e.target.value)}
              />
            </label>
          ) : (
            <label>
              Magnitude: <strong>{revenuePct}%</strong> revenue
              <input
                type="range"
                min={-50}
                max={0}
                step={1}
                value={revenuePct}
                onChange={(e) => setRevenuePct(+e.target.value)}
              />
            </label>
          )}
          {shockType === "sector_demand" && (
            <label>
              Sector
              <select value={sector} onChange={(e) => setSector(e.target.value)}>
                {SECTORS.map((s) => (
                  <option key={s} value={s}>
                    {s.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button className="btn accept" disabled={busy} onClick={run}>
            {busy ? "Re-scoring…" : "Run stress test"}
          </button>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <section className="score-card el-callout">
            <div className="el-figures">
              <div>
                <div className="pd-label">Pre-shock EL rate</div>
                <div className="pd-value">
                  {(result.pre_expected_loss_rate * 100).toFixed(2)}%
                </div>
              </div>
              <div className="el-arrow">→</div>
              <div>
                <div className="pd-label">Post-shock EL rate</div>
                <div className="pd-value">
                  {(result.post_expected_loss_rate * 100).toFixed(2)}%
                </div>
              </div>
              <div className={`el-delta ${deltaBps > 0 ? "worse" : "better"}`}>
                ΔEL {deltaBps >= 0 ? "+" : ""}
                {deltaBps.toFixed(1)} bps
              </div>
            </div>
            <div className="summary-meta">
              {result.n_loans_shocked.toLocaleString()} of{" "}
              {result.n_loans_scored.toLocaleString()} loans shocked · {result.sampling_note} ·{" "}
              {result.el_convention?.note}
            </div>
          </section>
          <div className="charts">
            <DistributionChart
              title="Pre-shock grade distribution"
              distribution={result.pre_shock_distribution}
              max={maxCount}
            />
            <DistributionChart
              title="Post-shock grade distribution"
              distribution={result.post_shock_distribution}
              max={maxCount}
            />
          </div>
        </>
      )}
    </div>
  );
}
