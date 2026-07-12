import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function HazardChart({ curve }) {
  if (!curve) {
    return (
      <div className="chart-card">
        <h3>12-month hazard curve</h3>
        <p className="chart-note">
          Monthly P(serious delinquency) — <em>when</em> risk concentrates, not just if.
        </p>
        <div className="point-estimate-note">
          Single-point estimate (TabPFN thin-file specialist) — this new-to-credit account has
          too little history for a month-by-month hazard curve, so the specialist model returns
          one honest 12-month PD instead.
        </div>
      </div>
    );
  }
  const data = curve.map((h, i) => ({ month: i + 1, hazard: +(h * 100).toFixed(3) }));
  return (
    <div className="chart-card">
      <h3>12-month hazard curve</h3>
      <p className="chart-note">
        Monthly P(serious delinquency) — <em>when</em> risk concentrates, not just if.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <defs>
            <linearGradient id="hazardFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#e05c1f" stopOpacity={0.7} />
              <stop offset="100%" stopColor="#e05c1f" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="month" label={{ value: "Month", position: "insideBottom", offset: -2 }} />
          <YAxis unit="%" width={55} />
          <Tooltip formatter={(v) => [`${v}%`, "hazard"]} labelFormatter={(m) => `Month ${m}`} />
          <Area type="monotone" dataKey="hazard" stroke="#e05c1f" fill="url(#hazardFill)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
