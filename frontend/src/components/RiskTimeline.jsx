import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const shortTs = (ts) => (ts || "").replace("T", " ").slice(5, 16);

export default function RiskTimeline({ timeline }) {
  const data = (timeline || []).map((t) => ({
    ts: shortTs(t.ts),
    pd: +(t.pd * 100).toFixed(2),
    grade: t.grade,
  }));
  return (
    <div className="chart-card">
      <h3>Risk trajectory</h3>
      <p className="chart-note">
        Re-scores from streaming events (GST filings, bank transactions, repayments) — silent
        moves land here; grade moves also raise an alert.
      </p>
      {data.length < 2 ? (
        <div className="point-estimate-note">
          {data.length === 0
            ? "No re-score history yet — simulate an event below to start the trajectory."
            : `One re-score so far (PD ${data[0].pd}%, grade ${data[0].grade}) — the line builds as more events arrive.`}
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="ts" tick={{ fontSize: 11 }} />
            <YAxis unit="%" width={55} />
            <Tooltip
              formatter={(v, _n, p) => [`${v}% (grade ${p.payload.grade})`, "calibrated PD"]}
            />
            <Line type="monotone" dataKey="pd" stroke="#5b8def" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
