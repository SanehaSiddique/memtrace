import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";
import { buildChains } from "../../graphUtils";
import MetricBadge from "../shared/MetricBadge";

// Predicate → short label, used to caption each lineage. Any predicate not
// listed still renders (raw, underscored) — this only prettifies the ones
// agents commonly emit.
const PREDICATE_LABELS = {
  uses_database: "Database",
  uses_auth: "Authentication",
  deployed_on: "Hosted on",
  depends_on: "Depends on",
  rejected: "Rejected",
};

const PIPELINE_STEPS = [
  "Query",
  "Retrieve candidates",
  "Follow relationships",
  "Check lifecycle status",
  "Exclude stale memory",
  "Build context",
  "Answer",
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Placeholders, not content. Nothing here asserts a fact about anyone's stack —
// the CEO types the real ones, which is the entire point of removing the seed.
const EXAMPLE_FACT = "We use Postgres as the primary datastore.";
const EXAMPLE_CHANGE = "We migrated the primary datastore to Postgres last week.";
const EXAMPLE_QUERY = "What are we using right now?";

export default function LiveMemoryDemo({ agentId, conversationId, onActivity }) {
  const [nodes, setNodes] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);

  const [eventText, setEventText] = useState("");
  const [simulating, setSimulating] = useState(false);
  const [animatingIds, setAnimatingIds] = useState({});

  const [queryText, setQueryText] = useState(EXAMPLE_QUERY);
  const [stage, setStage] = useState("idle"); // idle | querying | done
  const [stepIndex, setStepIndex] = useState(0);
  const [queryResult, setQueryResult] = useState(null);
  const [showTrace, setShowTrace] = useState(false);

  const [replaying, setReplaying] = useState(false);
  const [replayLog, setReplayLog] = useState([]);
  const [highlightedFromTimeline, setHighlightedFromTimeline] = useState(null);

  const byId = useMemo(() => Object.fromEntries(nodes.map((n) => [n.id, n])), [nodes]);

  async function reload() {
    const [graph, tl] = await Promise.all([api.getMemoryGraphFull(agentId), api.getMemoryTimeline(agentId)]);
    setNodes(graph.nodes);
    setTimeline(tl);
    return { nodes: graph.nodes, timeline: tl };
  }

  useEffect(() => {
    setLoading(true);
    reload().finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  // Every lineage, not just the three predicates the old demo story happened to
  // use. Filtering by a hardcoded predicate list would hide real memories the
  // agent extracted, which is the opposite of what this view is for.
  const chains = useMemo(() => buildChains(nodes), [nodes]);

  // Seeds ONE placeholder fact the CEO can edit or replace. This exists so the
  // graph isn't an empty box on a cold start — it is a starting point, not
  // fabricated content: whatever they type is what gets stored.
  async function handleStartFromExample() {
    if (eventText.trim()) {
      await handleSimulate();
      return;
    }
    setEventText(EXAMPLE_FACT);
    setQueryText(EXAMPLE_QUERY);
  }

  async function handleSimulate() {
    if (!eventText.trim() || simulating) return;
    setSimulating(true);
    try {
      const result = await api.ingest(eventText.trim(), agentId, conversationId);
      const op = result.operations[0];
      await reload();
      setAnimatingIds({ newId: op?.memory_id, oldId: op?.target_memory_id });
      setEventText("");
      setTimeout(() => setAnimatingIds({}), 2600);
      onActivity?.();
    } finally {
      setSimulating(false);
    }
  }

  async function handleRunQuery() {
    if (!queryText.trim() || stage === "querying") return;
    setStage("querying");
    setStepIndex(0);
    setQueryResult(null);
    setShowTrace(false);

    const resultPromise = api.debugQuery(queryText.trim(), agentId, conversationId);
    for (let i = 0; i < PIPELINE_STEPS.length; i++) {
      setStepIndex(i);
      // eslint-disable-next-line no-await-in-loop
      await sleep(420);
    }
    const result = await resultPromise;
    setQueryResult(result);
    setStage("done");
    onActivity?.();
  }

  async function handleReplay() {
    if (replaying || timeline.length === 0) return;
    setReplaying(true);
    setReplayLog([]);
    for (const m of timeline) {
      setHighlightedFromTimeline(m.id);
      const verb = m.supersedes_memory_id ? "updated" : "added";
      setReplayLog((log) => [
        ...log,
        { id: m.id, time: m.created_at, label: `${verb} → ${m.subject} ${m.predicate.replaceAll("_", " ")} ${m.object}` },
      ]);
      // eslint-disable-next-line no-await-in-loop
      await sleep(650);
    }
    setHighlightedFromTimeline(null);
    setReplaying(false);
  }

  function handleReset() {
    setStage("idle");
    setQueryResult(null);
    setShowTrace(false);
    setAnimatingIds({});
    setHighlightedFromTimeline(null);
    setReplayLog([]);
  }

  function nodeClassName(node) {
    const classes = ["memory-node"];
    if (node.status === "ACTIVE") classes.push("current");
    else if (node.status === "HISTORICAL") classes.push("historical");
    else if (node.status === "PENDING_REVIEW") classes.push("pending");
    else classes.push("historical");

    if (stage === "done" && queryResult) {
      if (queryResult.context.selected.some((sm) => sm.memory.id === node.id)) classes.push("pulse-active");
      if (queryResult.excluded.some((em) => em.memory.id === node.id)) classes.push("dim-excluded");
    }
    if (animatingIds.newId === node.id) classes.push("just-appeared", "pulse-active");
    if (animatingIds.oldId === node.id) classes.push("dim-superseded");
    if (highlightedFromTimeline === node.id) classes.push("pulse-active");
    return classes.join(" ");
  }

  if (loading) return <div className="empty-note">Loading live memory demo…</div>;

  if (chains.length === 0) {
    return (
      <div className="card">
        <h3 className="card-title">Live memory graph</h3>
        <div className="empty-note">
          Nothing remembered yet. Tell the agent one fact, then tell it a change — you'll watch the graph form, the old
          fact get retired, and the agent answer from what's still current.
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
          <input
            type="text"
            value={eventText}
            onChange={(e) => setEventText(e.target.value)}
            placeholder="e.g. We run our primary API on Kubernetes."
            style={{
              flex: 1,
              minWidth: 240,
              background: "var(--panel-2)",
              border: "1px solid var(--border)",
              color: "var(--text)",
              padding: "10px 12px",
              borderRadius: "var(--radius-sm)",
              fontSize: 13,
            }}
          />
          <button className="btn btn-primary" onClick={handleStartFromExample} disabled={simulating}>
            {simulating ? "Saving…" : "Store this fact"}
          </button>
          <button className="btn" onClick={() => setEventText(EXAMPLE_FACT)} title="Fill the field with an example you can edit">
            Use an example
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="card-title">
        Live memory graph <MetricBadge kind="ACTUAL" />
      </h3>
      <p className="card-note" style={{ marginTop: -6, marginBottom: 16 }}>
        Every node, edge, and answer below comes straight from the real MEMTRACE API — nothing here is drawn from
        fake state.
      </p>

      <div className="live-demo-controls">
        <button className="btn btn-ghost" onClick={handleReset} title="Resets this panel's view only — does not delete memory">
          Reset view
        </button>
        <button className="btn" onClick={handleReplay} disabled={replaying}>
          {replaying ? "Replaying…" : "Replay evolution"}
        </button>
      </div>

      {/* current state strip */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
        {chains.map((lineage) => {
          const head = lineage[0];
          return (
            <div key={head.id} className={nodeClassName(head)} style={{ cursor: "default" }}>
              {PREDICATE_LABELS[head.predicate] || head.predicate.replaceAll("_", " ")}: {head.object} {head.status === "ACTIVE" ? "✓" : `· ${head.status.toLowerCase()}`}
            </div>
          );
        })}
      </div>

      <div className="memory-flow">
        {chains.map((lineage) => {
          const head = lineage[0];
          const ancestors = lineage.slice(1).reverse();
          return (
            <div className="memory-chain" key={head.id}>
              <div style={{ fontSize: 12, color: "var(--text-faint)", marginBottom: 4 }}>
                {PREDICATE_LABELS[head.predicate] || head.predicate.replaceAll("_", " ")}
              </div>
              {ancestors.map((ancestor) => (
                <div key={ancestor.id}>
                  <div className={nodeClassName(ancestor)}>{ancestor.object}</div>
                  <div className="memory-arrow">↓ superseded</div>
                </div>
              ))}
              <div className={nodeClassName(head)}>
                {head.object} {head.status === "ACTIVE" ? "· current" : `· ${head.status.toLowerCase()}`}
              </div>
            </div>
          );
        })}
      </div>

      {replayLog.length > 0 && (
        <div className="trace-panel" style={{ marginTop: 16 }}>
          {replayLog.map((entry) => (
            <div className="trace-step" key={entry.id}>
              <div className="trace-label">{new Date(entry.time).toLocaleDateString()}</div>
              {entry.label}
            </div>
          ))}
        </div>
      )}

      <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "22px 0" }} />

      <div className="live-demo-controls">
        <input
          type="text"
          value={eventText}
          onChange={(e) => setEventText(e.target.value)}
          placeholder="Type a new fact for the agent to learn…"
        />
        <button className="btn btn-primary" onClick={handleSimulate} disabled={simulating}>
          {simulating ? "Simulating…" : "Simulate update"}
        </button>
      </div>

      <div className="live-demo-controls">
        <input
          type="text"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleRunQuery()}
          placeholder="Ask the agent anything about what it knows"
        />
        <button className="btn btn-primary" onClick={handleRunQuery} disabled={stage === "querying"}>
          {stage === "querying" ? "Running…" : "Run agent query"}
        </button>
      </div>

      {stage !== "idle" && (
        <div className="pipeline-steps">
          {PIPELINE_STEPS.map((label, i) => (
            <div
              key={label}
              className={`pipeline-step ${i === stepIndex && stage === "querying" ? "active" : i < stepIndex || stage === "done" ? "done" : ""}`}
            >
              {label}
            </div>
          ))}
        </div>
      )}

      {stage === "done" && queryResult && (
        <>
          <div className="answer-panel">
            <div className="q">{queryResult.query}</div>
            {queryResult.answer}
          </div>

          <div className="why-columns">
            <div className="why-box used">
              <h4>Why this memory was used</h4>
              <ul>
                {queryResult.context.selected.map((sm) => (
                  <li key={sm.memory.id}>
                    <strong>{sm.memory.object}</strong> — {sm.memory.status === "ACTIVE" ? "active, current" : sm.memory.status.toLowerCase()}
                    {sm.memory.supersedes_memory_id && byId[sm.memory.supersedes_memory_id]
                      ? `, supersedes ${byId[sm.memory.supersedes_memory_id].object}`
                      : ""}
                  </li>
                ))}
                {queryResult.context.selected.length === 0 && <li>No memory met the bar for this question.</li>}
              </ul>
            </div>
            <div className="why-box excluded">
              <h4>Why memory was excluded</h4>
              <ul>
                {queryResult.excluded.map((em) => (
                  <li key={em.memory.id}>
                    <strong>{em.memory.object}</strong> → {em.reason}
                  </li>
                ))}
                {queryResult.excluded.length === 0 && <li>Nothing was excluded for this question.</li>}
              </ul>
            </div>
          </div>

          <div className="cost-impact-panel">
            <div className="block headline">
              <div className="label">
                Cost impact <MetricBadge kind="CALCULATED" />
              </div>
              <div className="value">{formatUsd(queryResult.run_savings, { decimals: 4 })}</div>
              <div style={{ fontSize: 11, color: "var(--text-faint)" }}>unnecessary AI spend avoided on this run</div>
            </div>
            <div className="block">
              <div className="label">Memory decision</div>
              <div className="value">{queryResult.excluded.length}</div>
              <div style={{ fontSize: 11, color: "var(--text-faint)" }}>outdated/irrelevant memories excluded</div>
            </div>
            <div className="block">
              <div className="label">Context impact</div>
              <div className="value">
                {queryResult.context.selected.length}/{queryResult.retrieved.length}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                candidates kept · {queryResult.context.token_estimate} tokens sent
              </div>
            </div>
          </div>

          <button className="btn btn-ghost" style={{ marginTop: 14 }} onClick={() => setShowTrace((s) => !s)}>
            {showTrace ? "Hide trace" : "Why did the agent decide this?"}
          </button>

          {showTrace && <TracePanel result={queryResult} />}
        </>
      )}
    </div>
  );
}

function TracePanel({ result }) {
  const lifecycleLines = result.retrieved.map((sm) => {
    const excluded = result.excluded.find((e) => e.memory.id === sm.memory.id);
    const kept = result.context.selected.some((s) => s.memory.id === sm.memory.id);
    return `${sm.memory.object} (${sm.memory.status}) → ${excluded ? "excluded" : kept ? "kept" : "not used"}`;
  });

  return (
    <div className="trace-panel">
      <div className="trace-step">
        <div className="trace-label">Query</div>"{result.query}"
      </div>
      <div className="trace-step">
        <div className="trace-label">Retrieved memory IDs</div>
        <span className="mono" style={{ fontSize: 12 }}>
          {result.retrieved.map((sm) => `${sm.memory.id} (${sm.memory.object})`).join(", ") || "none"}
        </span>
      </div>
      <div className="trace-step">
        <div className="trace-label">Relationship traversal</div>
        {result.graph_paths.length > 0
          ? result.graph_paths
              .map((path) => path.map((step) => `${step.relationship.relation_type} → ${step.node_label}`).join(" "))
              .join(" · ")
          : "no graph edges traversed for this answer"}
      </div>
      <div className="trace-step">
        <div className="trace-label">Lifecycle decision</div>
        {lifecycleLines.join(" · ")}
      </div>
      <div className="trace-step">
        <div className="trace-label">Selected memory</div>
        {result.context.selected.map((sm) => sm.memory.object).join(", ") || "none"}
      </div>
      <div className="trace-step">
        <div className="trace-label">Final context sent to the model</div>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 11, color: "var(--text-dim)", margin: 0 }}>
          {result.context.context_text}
        </pre>
      </div>
      <div className="trace-step">
        <div className="trace-label">Answer</div>
        {result.answer}
      </div>
      {result.langsmith_run_id && (
        <div className="trace-step">
          <div className="trace-label">LangSmith run</div>
          <span className="mono">{result.langsmith_run_id}</span>
        </div>
      )}
    </div>
  );
}
