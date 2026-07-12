import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function DriverChart({ drivers }) {
  const data = drivers.map((d) => ({ ...d, name: d.label })).reverse();
  return (
    <div className="chart-card">
      <h3>Top risk drivers (SHAP)</h3>
      <p className="chart-note">Verified model attributions — the source of truth for every explanation.</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" width={190} tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(v, _n, p) => [
              `${v > 0 ? "+" : ""}${v} (${p.payload.direction}, value=${p.payload.value})`,
              "SHAP",
            ]}
          />
          <ReferenceLine x={0} stroke="#888" />
          <Bar dataKey="shap">
            {data.map((d) => (
              <Cell key={d.feature} fill={d.shap > 0 ? "#d43a2f" : "#2e9e5b"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
