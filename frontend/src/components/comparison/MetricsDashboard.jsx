import { formatNumber, formatUsd } from "../../format";
import AnimatedNumber from "../shared/AnimatedNumber";

/**
 * Side-by-side metric cards/bars per TurnMetrics field (docs/IMPLEMENTATION.md §9).
 * Displays live per-turn metrics and running cumulative totals.
 */
export default function MetricsDashboard({ latestA1Metrics, latestA2Metrics, sessionSummary }) {
  const savings = sessionSummary?.savings || {
    tokens_saved: 0,
    tokens_saved_pct: 0,
    llm_calls_saved: 0,
    cost_saved_gpt4o: 0,
    cost_saved_claude: 0,
  };

  const a1 = sessionSummary?.agent1 || {
    turns_count: 0,
    total_context_tokens: 0,
    total_llm_calls: 0,
    projected_cost_gpt4o: 0,
    projected_cost_claude: 0,
  };

  const a2 = sessionSummary?.agent2 || {
    turns_count: 0,
    total_context_tokens: 0,
    total_llm_calls: 0,
    total_jev_calls: 0,
    projected_cost_gpt4o: 0,
    projected_cost_claude: 0,
  };

  return (
    <div className="metrics-dashboard-root">
      {/* Cumulative KPI Highlights */}
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="card stat-card stat-card-highlight">
          <div className="stat-label">Total Prompt Tokens Saved</div>
          <div className="stat-number stat-green">
            <AnimatedNumber value={savings.tokens_saved} format={(v) => formatNumber(Math.round(v))} />
          </div>
          <div className="card-note">
            {savings.tokens_saved_pct > 0 ? `${savings.tokens_saved_pct}% total token reduction` : "Awaiting first query run"}
          </div>
        </div>

        <div className="card stat-card">
          <div className="stat-label">LLM Calls Avoided</div>
          <div className="stat-number stat-blue">
            <AnimatedNumber value={savings.llm_calls_saved} format={(v) => formatNumber(Math.round(v))} />
          </div>
          <div className="card-note">Fast-path JEV decisions & routing</div>
        </div>

        <div className="card stat-card">
          <div className="stat-label">Projected GPT-4o Savings</div>
          <div className="stat-number stat-green">
            <AnimatedNumber value={savings.cost_saved_gpt4o} format={(v) => formatUsd(v, { decimals: 4 })} />
          </div>
          <div className="card-note">Cumulative spend prevented</div>
        </div>

        <div className="card stat-card">
          <div className="stat-label">JEV Fast-Evaluations</div>
          <div className="stat-number stat-purple">
            <AnimatedNumber value={a2.total_jev_calls} format={(v) => formatNumber(Math.round(v))} />
          </div>
          <div className="card-note">Tool routing + chunk scoring + staleness</div>
        </div>
      </div>

      {/* Side-by-side comparison table */}
      <div className="card" style={{ marginBottom: 20 }}>
        <h3 className="card-title">Side-by-Side Turn Metrics (Latest Turn)</h3>
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th style={{ color: "var(--text-dim)" }}>Agent 1 (Postgres + Raw Tool)</th>
                <th style={{ color: "var(--green)" }}>Agent 2 (Neo4j Graph + JEV Filter)</th>
                <th style={{ color: "var(--amber)" }}>Delta / Advantage</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Context / Prompt Tokens</strong></td>
                <td>{latestA1Metrics ? formatNumber(latestA1Metrics.total_context_tokens) : "—"}</td>
                <td>
                  <strong>{latestA2Metrics ? formatNumber(latestA2Metrics.total_context_tokens) : "—"}</strong>
                </td>
                <td style={{ color: "var(--green)" }}>
                  {latestA1Metrics && latestA2Metrics
                    ? `-${Math.max(0, latestA1Metrics.total_context_tokens - latestA2Metrics.total_context_tokens)} tok (${
                        latestA1Metrics.total_context_tokens > 0
                          ? Math.round(((latestA1Metrics.total_context_tokens - latestA2Metrics.total_context_tokens) / latestA1Metrics.total_context_tokens) * 100)
                          : 0
                      }%)`
                    : "—"}
                </td>
              </tr>
              <tr>
                <td><strong>Tool Result Chunks / Filter</strong></td>
                <td>
                  {latestA1Metrics?.raw_result_tokens ? `${latestA1Metrics.raw_result_tokens} tok (unfiltered)` : "No tool call"}
                </td>
                <td>
                  {latestA2Metrics?.raw_result_tokens
                    ? `${latestA2Metrics.raw_result_tokens} → ${latestA2Metrics.filtered_result_tokens || 0} tok (filtered)`
                    : "Filtered or skipped"}
                </td>
                <td style={{ color: "var(--green)" }}>
                  {latestA2Metrics?.raw_result_tokens && latestA2Metrics?.filtered_result_tokens
                    ? `-${latestA2Metrics.raw_result_tokens - latestA2Metrics.filtered_result_tokens} tok saved`
                    : "—"}
                </td>
              </tr>
              <tr>
                <td><strong>Tools In Prompt</strong></td>
                <td>{latestA1Metrics ? `${latestA1Metrics.tools_considered} schemas` : "—"}</td>
                <td>{latestA2Metrics ? `${latestA2Metrics.tools_considered} schemas` : "—"}</td>
                <td style={{ color: "var(--blue)" }}>
                  {latestA1Metrics && latestA2Metrics && latestA1Metrics.tools_considered > latestA2Metrics.tools_considered
                    ? `${latestA1Metrics.tools_considered - latestA2Metrics.tools_considered} schemas pruned by JEV`
                    : "Identical"}
                </td>
              </tr>
              <tr>
                <td><strong>Total Turn Latency</strong></td>
                <td>{latestA1Metrics ? `${Math.round(latestA1Metrics.latency_ms_total)}ms` : "—"}</td>
                <td>{latestA2Metrics ? `${Math.round(latestA2Metrics.latency_ms_total)}ms` : "—"}</td>
                <td>
                  {latestA1Metrics && latestA2Metrics
                    ? `${Math.round(latestA2Metrics.latency_ms_total - latestA1Metrics.latency_ms_total)}ms`
                    : "—"}
                </td>
              </tr>
              <tr>
                <td><strong>Est. Turn Cost (GPT-4o)</strong></td>
                <td>
                  {latestA1Metrics?.cost_projected?.["gpt-4o"] !== undefined
                    ? formatUsd(latestA1Metrics.cost_projected["gpt-4o"], { decimals: 4 })
                    : "—"}
                </td>
                <td>
                  {latestA2Metrics?.cost_projected?.["gpt-4o"] !== undefined
                    ? formatUsd(latestA2Metrics.cost_projected["gpt-4o"], { decimals: 4 })
                    : "—"}
                </td>
                <td style={{ color: "var(--green)" }}>
                  {latestA1Metrics?.cost_projected?.["gpt-4o"] !== undefined &&
                  latestA2Metrics?.cost_projected?.["gpt-4o"] !== undefined
                    ? formatUsd(
                        Math.max(
                          0,
                          latestA1Metrics.cost_projected["gpt-4o"] - latestA2Metrics.cost_projected["gpt-4o"]
                        ),
                        { decimals: 4 }
                      ) + " saved"
                    : "—"}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
