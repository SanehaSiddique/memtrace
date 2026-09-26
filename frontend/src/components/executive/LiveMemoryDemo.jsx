import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";
import { buildChains } from "../../graphUtils";
import MetricBadge from "../shared/MetricBadge";

const FOCUS_PREDICATES = {
  uses_database: "Database",
  uses_auth: "Authentication",
  deployed_on: "Hosted on",
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

function suggestNextEvent(memories) {
  const objects = memories.map((m) => m.object);
  const contents = memories.map((m) => m.content);
  if (!objects.includes("PostgreSQL")) {
    return "We migrated from MongoDB to PostgreSQL because relational querying became important.";
  }
  if (!contents.some((c) => c.includes("connection pool"))) {
    return "We updated the PostgreSQL configuration to increase the connection pool size.";
  }
  return "";
}

export default function LiveMemoryDemo({ agentId, conversationId, onActivity }) {
  const [nodes, setNodes] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);

  const [eventText, setEventText] = useState("");
  const [simulating, setSimulating] = useState(false);
  const [animatingIds, setAnimatingIds] = useState({});

  const [queryText, setQueryText] = useState("Which database does Project Alpha use?");
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
    reload()
      .then(({ nodes: n }) => setEventText(suggestNextEvent(n)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  const chains = useMemo(
    () => buildChains(nodes).filter((lineage) => FOCUS_PREDICATES[lineage[0].predicate]),
    [nodes],
  );

  // A focused 3-fact starting point (matching the classic "MongoDB / session
  // auth / AWS, all ACTIVE" scenario) — deliberately smaller than the full
  // "Seed demo data" story in the header, so the MongoDB -> PostgreSQL
  // migration below is something you can actually watch happen live, not
  // something that already happened before you opened the page. If the
  // header's full seed already ran, `chains.length > 0` and this never shows.
  async function handleSetUpScenario() {
    setLoading(true);
    try {
      await api.ingest("Project Alpha uses MongoDB.", agentId, conversationId);
      await api.ingest("Project Alpha uses session-based authentication.", agentId, conversationId);
      await api.ingest("Project Alpha uses AWS.", agentId, conversationId);
      const { nodes: n } = await reload();
      setEventText(suggestNextEvent(n));
      onActivity?.();
    } finally {
      setLoading(false);
    }
  }

  async function handleSimulate() {
    if (!eventText.trim() || simulating) return;
    setSimulating(true);
    try {
      const result = await api.ingest(eventText.trim(), agentId, conversationId);
      const op = result.operations[0];
      const { nodes: n } = await reload();
      setAnimatingIds({ newId: op?.memory_id, oldId: op?.target_memory_id });
      setEventText(suggestNextEvent(n));
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
          Set up the scenario to see MongoDB → PostgreSQL migrate live (this is a smaller starting point than the
          header's "Seed demo data", so you can actually watch the migration happen below).
        </div>
        <button className="btn btn-primary" onClick={handleSetUpScenario} style={{ marginTop: 12 }}>
          Set up scenario
        </button>
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
              {FOCUS_PREDICATES[head.predicate]}: {head.object} {head.status === "ACTIVE" ? "✓" : `· ${head.status.toLowerCase()}`}
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
                {FOCUS_PREDICATES[head.predicate]}
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
          placeholder='Ask e.g. "Which database does Project Alpha use?"'
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
