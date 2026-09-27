import { formatUsd } from "../../format";

/**
 * Cross-model cost comparison bar chart (docs/IMPLEMENTATION.md §9).
 * Agent1 vs Agent2 side by side across models (GPT-4o, Claude Sonnet 3.5, Grok).
 */
export default function CostComparison({ sessionSummary, latestA1Metrics, latestA2Metrics }) {
  const models = [
    { key: "gpt-4o", label: "GPT-4o" },
    { key: "claude-sonnet", label: "Claude Sonnet 3.5" },
    { key: "grok", label: "Grok 2" },
  ];

  const a1Costs = latestA1Metrics?.cost_projected || {};
  const a2Costs = latestA2Metrics?.cost_projected || {};

  const summary = sessionSummary || {};
  const cumA1 = summary.agent1 || {};
  const cumA2 = summary.agent2 || {};

  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="section-heading">
        <div>
          <h3 className="card-title" style={{ margin: 0 }}>Projected Cost Comparison Across Models</h3>
          <p className="card-note" style={{ margin: "4px 0 0 0" }}>
            Real prompt and completion token counts multiplied by standard commercial pricing per model.
          </p>
        </div>
      </div>

      <div className="cost-bars-container" style={{ marginTop: 16 }}>
        {models.map((m) => {
          const a1Val = a1Costs[m.key] || 0;
          const a2Val = a2Costs[m.key] || 0;
          const maxVal = Math.max(0.0001, a1Val, a2Val);
          const a1Pct = Math.round((a1Val / maxVal) * 100);
          const a2Pct = Math.round((a2Val / maxVal) * 100);
          const saved = Math.max(0, a1Val - a2Val);

          return (
            <div key={m.key} className="cost-model-row" style={{ marginBottom: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 14 }}>{m.label}</span>
                {saved > 0 && (
                  <span style={{ color: "var(--green)", fontSize: 13, fontWeight: 600 }}>
                    Savings: {formatUsd(saved, { decimals: 4 })} / turn
                  </span>
                )}
              </div>

              {/* Agent 1 Bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                <span style={{ width: 65, fontSize: 12, color: "var(--text-dim)" }}>Agent 1</span>
                <div style={{ flex: 1, background: "var(--panel-2)", height: 18, borderRadius: 4, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${a1Pct}%`,
                      height: "100%",
                      background: "var(--red)",
                      borderRadius: 4,
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
                <span style={{ width: 85, fontSize: 12, textAlign: "right", fontFamily: "monospace" }}>
                  {formatUsd(a1Val, { decimals: 4 })}
                </span>
              </div>

              {/* Agent 2 Bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ width: 65, fontSize: 12, color: "var(--green)" }}>Agent 2</span>
                <div style={{ flex: 1, background: "var(--panel-2)", height: 18, borderRadius: 4, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${a2Pct}%`,
                      height: "100%",
                      background: "var(--green)",
                      borderRadius: 4,
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
                <span style={{ width: 85, fontSize: 12, textAlign: "right", fontFamily: "monospace", color: "var(--green)" }}>
                  {formatUsd(a2Val, { decimals: 4 })}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Cumulative Session Summary Footer */}
      {(cumA1.projected_cost_gpt4o > 0 || cumA2.projected_cost_gpt4o > 0) && (
        <div style={{ borderTop: "1px solid var(--border)", paddingTop: 14, marginTop: 14, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
          <div>
            <span style={{ color: "var(--text-dim)", fontSize: 12 }}>Cumulative Session (GPT-4o):</span>
            <div style={{ fontSize: 15, fontWeight: 700, marginTop: 2 }}>
              Agent 1: <span style={{ color: "var(--red)" }}>{formatUsd(cumA1.projected_cost_gpt4o, { decimals: 4 })}</span>
              {" vs "}
              Agent 2: <span style={{ color: "var(--green)" }}>{formatUsd(cumA2.projected_cost_gpt4o, { decimals: 4 })}</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <span style={{ color: "var(--text-dim)", fontSize: 12 }}>Net Spend Prevented:</span>
            <div style={{ fontSize: 16, fontWeight: 800, color: "var(--green)", marginTop: 2 }}>
              {formatUsd(Math.max(0, (cumA1.projected_cost_gpt4o || 0) - (cumA2.projected_cost_gpt4o || 0)), { decimals: 4 })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
