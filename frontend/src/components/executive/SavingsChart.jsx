import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";

const GRANULARITIES = [
  { key: "day", label: "Daily" },
  { key: "week", label: "Weekly" },
  { key: "month", label: "Monthly" },
  { key: "year", label: "Annual" },
];

export default function SavingsChart({ agentId, refreshSignal }) {
  const [granularity, setGranularity] = useState("day");
  const [points, setPoints] = useState([]);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    api.costTimeseries(agentId, granularity).then(setPoints).catch(() => setPoints([]));
  }, [agentId, granularity, refreshSignal]);

  return (
    <div>
      <div className="tabs" style={{ marginBottom: 14 }}>
        {GRANULARITIES.map((g) => (
          <button key={g.key} className={g.key === granularity ? "active" : ""} onClick={() => setGranularity(g.key)}>
            {g.label}
          </button>
        ))}
      </div>
      <Chart points={points} hover={hover} setHover={setHover} />
    </div>
  );
}

function Chart({ points, hover, setHover }) {
  if (!points || points.length === 0) {
    return <div className="empty-note">No savings history yet — ask a few questions to start the curve.</div>;
  }

  const width = 720;
  const height = 220;
  const padding = { top: 10, right: 10, bottom: 10, left: 10 };
  const maxY = Math.max(...points.map((p) => p.cumulative_savings), 0.01);
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const xFor = (i) => padding.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
  const yFor = (v) => padding.top + innerH - (v / maxY) * innerH;

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(i).toFixed(1)} ${yFor(p.cumulative_savings).toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L ${xFor(points.length - 1).toFixed(1)} ${(padding.top + innerH).toFixed(1)} L ${xFor(0).toFixed(1)} ${(padding.top + innerH).toFixed(1)} Z`;

  return (
    <div className="chart-wrap">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ minWidth: 480, display: "block" }}>
        <defs>
          <linearGradient id="savingsFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3ddc84" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#3ddc84" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#savingsFill)" />
        <path d={linePath} fill="none" stroke="#3ddc84" strokeWidth="2.5" />
        {points.map((p, i) => (
          <circle
            key={p.date}
            cx={xFor(i)}
            cy={yFor(p.cumulative_savings)}
            r={hover === i ? 5 : 3}
            fill="#3ddc84"
            style={{ cursor: "pointer" }}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
      <div className="chart-caption">
        {hover !== null
          ? `${points[hover].date}: ${formatUsd(points[hover].cumulative_savings)} cumulative (${formatUsd(points[hover].daily_savings)} that period)`
          : `Latest: ${formatUsd(points[points.length - 1].cumulative_savings)} saved across ${points.length} period(s) of activity`}
      </div>
      <p className="card-note">
        Every avoided piece of unnecessary context compounds across every agent, every conversation, every day.
      </p>
    </div>
  );
}
