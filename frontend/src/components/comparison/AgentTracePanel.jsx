import { useState } from "react";

/**
 * Live "thinking process" timeline for one agent (docs/IMPLEMENTATION_V2.md §5.3).
 * Renders the TraceEvent stream for the current turn as a readable sequence, not
 * a log dump. Agent1's trace is intentionally sparser than Agent2's — no jev_*
 * steps exist for it — and that asymmetry is the point, not a bug to paper over.
 */

const STEP_LABELS = {
  stm_loaded: "Loaded memory",
  jev_routing_start: "JEV routing tool selection",
  jev_routing_result: "JEV routing decision",
  llm_reasoning_start: "LLM reasoning",
  tool_call_start: "Calling tool",
  tool_call_result: "Tool result received",
  jev_filtering_start: "JEV filtering tool result",
  jev_filtering_result: "JEV filtering decision",
  llm_final_answer_start: "Generating final answer",
  graph_write_start: "Background: writing to graph",
};

// Rendered inline within the timeline row instead (or not rendered at all), not
// as their own generic row.
const SKIP_STEPS = new Set(["llm_final_answer_token", "turn_complete", "graph_updated"]);

function ProbabilityBars({ probabilities, chosen }) {
  const entries = Object.entries(probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  if (entries.length === 0) return null;
  const max = Math.max(0.01, ...entries.map(([, p]) => p));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
      {entries.map(([tool, prob]) => {
        const isChosen = tool === chosen;
        return (
          <div key={tool} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                fontSize: 11,
                fontFamily: "monospace",
                width: 160,
                flexShrink: 0,
                color: isChosen ? "var(--green)" : "var(--text-dim)",
                fontWeight: isChosen ? 700 : 400,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
              title={tool}
            >
              {tool}
            </span>
            <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div
                style={{
                  width: `${(prob / max) * 100}%`,
                  height: "100%",
                  background: isChosen ? "var(--green)" : "var(--border)",
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <span style={{ fontSize: 11, color: "var(--text-faint)", width: 34, textAlign: "right" }}>
              {prob.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function FilterChunks({ chunks }) {
  if (!chunks || chunks.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
      {chunks.map((c, idx) => (
        <div
          key={idx}
          style={{
            fontSize: 11,
            fontFamily: "monospace",
            padding: "4px 8px",
            borderRadius: 4,
            background: c.kept ? "var(--green-dim)" : "rgba(255,107,107,0.06)",
            color: c.kept ? "var(--text)" : "var(--text-faint)",
            textDecoration: c.kept ? "none" : "line-through",
            display: "flex",
            justifyContent: "space-between",
            gap: 8,
          }}
        >
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {c.text_preview}
          </span>
          <span style={{ flexShrink: 0, color: "var(--text-dim)" }}>{c.score != null ? c.score.toFixed(2) : "—"}</span>
        </div>
      ))}
    </div>
  );
}

function TraceStep({ event }) {
  const { step, detail } = event;
  const label = STEP_LABELS[step] || step;
  let extra = null;

  if (step === "jev_routing_start") {
    extra = (
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
        Considering {detail.tool_options_count} tools for: "{detail.query}"
      </div>
    );
  } else if (step === "jev_routing_result") {
    const count = Object.keys(detail.all_probabilities || {}).length;
    extra = (
      <>
        <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
          Chose <strong style={{ color: "var(--green)" }}>{detail.chosen_tool}</strong> from {count} candidates (
          {Math.round(detail.latency_ms || 0)}ms)
        </div>
        <ProbabilityBars probabilities={detail.all_probabilities} chosen={detail.chosen_tool} />
      </>
    );
  } else if (step === "llm_reasoning_start") {
    extra = (
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
        {detail.tools_in_prompt} tool schema(s) in prompt · {detail.model}
      </div>
    );
  } else if (step === "tool_call_start") {
    extra = <div style={{ fontSize: 12, color: "var(--text-dim)" }}>{detail.tool_name}</div>;
  } else if (step === "tool_call_result") {
    extra = <div style={{ fontSize: 12, color: "var(--text-dim)" }}>~{detail.raw_tokens_estimate} raw tokens</div>;
  } else if (step === "jev_filtering_start") {
    extra = <div style={{ fontSize: 12, color: "var(--text-dim)" }}>{detail.chunk_count} chunk(s) to score</div>;
  } else if (step === "jev_filtering_result") {
    const kept = (detail.chunks || []).filter((c) => c.kept).length;
    extra = (
      <>
        <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
          Kept {kept}/{(detail.chunks || []).length} chunks, ~{detail.filtered_tokens_estimate} tokens
        </div>
        <FilterChunks chunks={detail.chunks} />
      </>
    );
  } else if (step === "stm_loaded") {
    extra = <div style={{ fontSize: 12, color: "var(--text-dim)" }}>{detail.message_count} prior message(s)</div>;
  }

  return (
    <div style={{ padding: "6px 0", borderBottom: "1px dashed var(--border)" }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)" }}>{label}</div>
      {extra}
    </div>
  );
}

export default function AgentTracePanel({ agentId, events = [] }) {
  const [collapsed, setCollapsed] = useState(false);
  const visible = events.filter((e) => !SKIP_STEPS.has(e.step));

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
        onClick={() => setCollapsed((c) => !c)}
      >
        <h4 style={{ margin: 0, fontSize: 13, color: "var(--text-dim)", textTransform: "uppercase" }}>
          Live Trace — {agentId === "agent1" ? "Agent 1" : "Agent 2"}
        </h4>
        <span style={{ fontSize: 11, color: "var(--text-faint)" }}>{collapsed ? "Show" : "Hide"}</span>
      </div>
      {!collapsed && (
        <div style={{ marginTop: 8, maxHeight: 320, overflowY: "auto" }}>
          {visible.length === 0 ? (
            <div style={{ fontSize: 12, color: "var(--text-faint)" }}>No activity yet this turn.</div>
          ) : (
            visible.map((e, idx) => <TraceStep key={idx} event={e} />)
          )}
        </div>
      )}
    </div>
  );
}
