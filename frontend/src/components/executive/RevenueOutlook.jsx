import { useEffect, useMemo, useState } from "react";
import { api, DEFAULT_AGENT_ID } from "../../api";
import { formatCompactUsd, formatNumber, formatUsd } from "../../format";
import MetricBadge from "../shared/MetricBadge";

/**
 * CEO-perspective cost & revenue outlook.
 *
 * Nothing here is a user-typed average. The token volumes come from the real
 * recorded runs (`/cost/revenue-projection` returns a `profile` saying whether
 * they are measured or a documented fallback), and the rate is each model's
 * published price. So moving the agent slider or swapping OpenAI for Claude
 * changes the number for a reason you can trace, and the delta column is
 * arithmetic between two real figures rather than a percentage of a guess.
 */

const PROVIDER_LABELS = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  grok: "Grok (xAI)",
  groq: "Groq",
  openrouter: "OpenRouter",
};

const PROVIDER_COLORS = {
  openai: "#10a37f",
  anthropic: "#d97757",
  grok: "#e5e5e5",
  groq: "#f55036",
  openrouter: "#6467f2",
};

function DeltaCell({ value }) {
  if (!value) return <span className="rev-delta same">—</span>;
  const cls = value < 0 ? "cheaper" : "dearer";
  return (
    <span className={`rev-delta ${cls}`}>
      {value < 0 ? "▼" : "▲"} {formatCompactUsd(Math.abs(value))}
    </span>
  );
}

export default function RevenueOutlook({ refreshSignal = 0 }) {
  const [agents, setAgents] = useState(100);
  const [runsPerDay, setRunsPerDay] = useState(500);
  const [revenuePerAgent, setRevenuePerAgent] = useState(800);
  const [currentModel, setCurrentModel] = useState("gpt-4o");
  const [catalog, setCatalog] = useState({});
  const [projection, setProjection] = useState(null);

  useEffect(() => {
    api.costModels().then(setCatalog).catch(() => {});
  }, []);

  useEffect(() => {
    const handle = setTimeout(() => {
      api
        .revenueProjection({
          agent_id: DEFAULT_AGENT_ID,
          agents: Number(agents),
          runs_per_agent_per_day: Number(runsPerDay),
          current_model: currentModel,
          revenue_per_agent_month: Number(revenuePerAgent),
        })
        .then(setProjection)
        .catch(() => {});
    }, 120);
    return () => clearTimeout(handle);
  }, [agents, runsPerDay, currentModel, revenuePerAgent, refreshSignal]);

  const currentRow = useMemo(
    () => projection?.rows.find((r) => r.model === currentModel) || projection?.rows[0] || null,
    [projection, currentModel]
  );

  const providerGroups = useMemo(
    () => Object.entries(catalog).map(([provider, models]) => ({ provider, label: PROVIDER_LABELS[provider] || provider, models })),
    [catalog]
  );

  if (!projection || !currentRow) {
    return <div className="empty-note">Loading revenue outlook…</div>;
  }

  const { profile, best_model: bestModel, best_model_annual_savings: bestSavings } = projection;
  const memorySavingsPct = currentRow.savings_pct;


  return (
    <div>
      <div className="rev-headline">
        <div className="figure">{formatCompactUsd(currentRow.annual_savings)}</div>
        <div className="caption">
          annual AI spend avoided at {projection.agents.toLocaleString()} agents on {currentRow.label}
          <MetricBadge kind={profile.is_measured ? "CALCULATED" : "PROJECTED"} />
        </div>
      </div>

      <div className="rev-kpis">
        <div className="rev-kpi">
          <div className="label">Weekly</div>
          <div className="value">{formatCompactUsd(currentRow.weekly_savings)}</div>
          <div className="sub">saved / week</div>
        </div>
        <div className="rev-kpi">
          <div className="label">Monthly</div>
          <div className="value">{formatCompactUsd(currentRow.monthly_savings)}</div>
          <div className="sub">saved / month</div>
        </div>
        <div className="rev-kpi">
          <div className="label">Annual</div>
          <div className="value">{formatCompactUsd(currentRow.annual_savings)}</div>
          <div className="sub">saved / year</div>
        </div>
        <div className="rev-kpi">
          <div className="label">Context reduction</div>
          <div className="value">{memorySavingsPct.toFixed(1)}%</div>
          <div className="sub">
            {formatNumber(profile.baseline_input_tokens)} → {formatNumber(profile.optimized_input_tokens)} input tokens
          </div>
        </div>
      </div>

      {bestModel && (
        <div className="impact-line" style={{ background: "var(--green-dim)", color: "var(--green)", marginBottom: 18 }}>
          Switching {currentRow.label} → {bestModel} at this scale saves{" "}
          <strong>{formatCompactUsd(bestSavings)}/year</strong>
          {projection.margin_points_recovered > 0 && (
            <> — {projection.margin_points_recovered.toFixed(2)} margin points at ${Number(revenuePerAgent).toLocaleString()} revenue/agent/month</>
          )}
        </div>
      )}

      <div className="rev-grid">
        <div className="rev-controls">
          <div className="rev-field">
            <label>
              Number of agents <span className="val">{Number(agents).toLocaleString()}</span>
            </label>
            <input type="range" min="1" max="5000" step="1" value={agents} onChange={(e) => setAgents(e.target.value)} />
          </div>
          <div className="rev-field">
            <label>
              Runs / agent / day <span className="val">{Number(runsPerDay).toLocaleString()}</span>
            </label>
            <input type="range" min="1" max="5000" step="1" value={runsPerDay} onChange={(e) => setRunsPerDay(e.target.value)} />
          </div>
          <div className="rev-field">
            <label>
              Revenue / agent / month <span className="val">${Number(revenuePerAgent).toLocaleString()}</span>
            </label>
            <input
              type="number"
              min="0"
              step="50"
              value={revenuePerAgent}
              onChange={(e) => setRevenuePerAgent(e.target.value)}
            />
          </div>
          <div className="rev-field">
            <label>Current model</label>
            <select value={currentModel} onChange={(e) => setCurrentModel(e.target.value)}>
              {providerGroups.map((group) => (
                <optgroup key={group.provider} label={group.label}>
                  {group.models.map((m) => (
                    <option key={m.model} value={m.model}>
                      {m.label || m.model}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
        </div>

        <div>
          <div className="comparison-table-wrapper">
            <table className="rev-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Weekly</th>
                  <th>Monthly</th>
                  <th>Annual</th>
                  <th>vs current (annual)</th>
                </tr>
              </thead>
              <tbody>
                {projection.rows.map((row) => {
                  const isCurrent = row.model === currentModel;
                  const isBest = row.model === bestModel;
                  return (
                    <tr key={row.model} className={`${isCurrent ? "is-current" : ""} ${isBest ? "is-best" : ""}`}>
                      <td>
                        <span className="rev-model-name">
                          <span
                            className="class-swatch"
                            style={{ background: PROVIDER_COLORS[row.provider] || "var(--text-faint)" }}
                          />
                          {row.label}
                          <span className="rev-provider-pill">{row.provider}</span>
                          {isBest && <span className="badge badge-active">cheapest</span>}
                        </span>
                      </td>
                      <td>{formatCompactUsd(row.weekly_with_memtrace)}</td>
                      <td>{formatCompactUsd(row.monthly_with_memtrace)}</td>
                      <td>{formatCompactUsd(row.annual_with_memtrace)}</td>
                      <td>
                        {isCurrent ? (
                          <span className="rev-delta same">current</span>
                        ) : (
                          <DeltaCell value={row.annual_delta_vs_current} />
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="rev-source-note">
            {projection.data_source_note} Spend columns are <em>with</em> MEMTRACE; the headline figure is the
            context it avoided at the measured {memorySavingsPct.toFixed(1)}% reduction (
            {formatUsd(currentRow.monthly_gross - currentRow.monthly_with_memtrace, { decimals: 2 })}/mo).
          </div>
        </div>
      </div>
    </div>
  );
}
