import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";

export default function ScaleCalculator() {
  const [agents, setAgents] = useState(100);
  const [runsPerDay, setRunsPerDay] = useState(500);
  const [costPerRun, setCostPerRun] = useState(0.08);
  const [avoidable, setAvoidable] = useState(22);
  const [projection, setProjection] = useState(null);

  useEffect(() => {
    const handle = setTimeout(() => {
      api
        .scaleProjection({
          agents: Number(agents),
          runs_per_agent_per_day: Number(runsPerDay),
          cost_per_run: Number(costPerRun),
          avoidable_pct: Number(avoidable),
        })
        .then(setProjection)
        .catch(() => {});
    }, 100);
    return () => clearTimeout(handle);
  }, [agents, runsPerDay, costPerRun, avoidable]);

  return (
    <div className="calc-grid">
      <div>
        <div className="calc-field">
          <label>
            Number of agents <span className="val">{Number(agents).toLocaleString()}</span>
          </label>
          <input type="range" min="1" max="5000" value={agents} onChange={(e) => setAgents(e.target.value)} />
        </div>
        <div className="calc-field">
          <label>
            Average runs per agent / day <span className="val">{Number(runsPerDay).toLocaleString()}</span>
          </label>
          <input type="range" min="1" max="5000" value={runsPerDay} onChange={(e) => setRunsPerDay(e.target.value)} />
        </div>
        <div className="calc-field">
          <label>
            Average AI cost / run <span className="val">${Number(costPerRun).toFixed(3)}</span>
          </label>
          <input
            type="range"
            min="0.001"
            max="1"
            step="0.001"
            value={costPerRun}
            onChange={(e) => setCostPerRun(e.target.value)}
          />
        </div>
        <div className="calc-field">
          <label>
            Estimated avoidable context cost <span className="val">{avoidable}%</span>
          </label>
          <input type="range" min="0" max="60" value={avoidable} onChange={(e) => setAvoidable(e.target.value)} />
        </div>
      </div>

      <div className="calc-results">
        <div className="calc-result">
          <span className="label">Daily savings</span>
          <span className="value">{projection ? formatUsd(projection.daily_savings) : "—"}</span>
        </div>
        <div className="calc-result">
          <span className="label">Monthly savings</span>
          <span className="value">{projection ? formatUsd(projection.monthly_savings) : "—"}</span>
        </div>
        <div className="calc-result annual">
          <span className="label">Annual savings</span>
          <span className="value">{projection ? formatUsd(projection.annual_savings, { decimals: 0 }) : "—"}</span>
        </div>
      </div>
    </div>
  );
}
