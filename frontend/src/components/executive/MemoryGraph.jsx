import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";
import { buildChains, edgeRelationLabel } from "../../graphUtils";

export default function MemoryGraph({ agentId, avgSavingsPerRun, refreshSignal }) {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selected, setSelected] = useState(null);
  const [impact, setImpact] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .getMemoryGraphFull(agentId)
      .then((g) => {
        setNodes(g.nodes);
        setEdges(g.edges);
      })
      .catch(() => {
        setNodes([]);
        setEdges([]);
      })
      .finally(() => setLoading(false));
  }, [agentId, refreshSignal]);

  useEffect(() => {
    if (!selected) {
      setImpact(null);
      return;
    }
    api
      .getMemoryCostImpact(selected.id)
      .then((r) => setImpact(r.cost_avoided))
      .catch(() => setImpact(null));
  }, [selected]);

  if (loading) return <div className="empty-note">Loading memory graph…</div>;
  if (nodes.length === 0) {
    return <div className="empty-note">No memory yet — seed demo data to see your agent's memory here.</div>;
  }

  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const chains = buildChains(nodes);
  const edgeBetween = (oldId, newId) => edgeRelationLabel(edges, oldId, newId);

  return (
    <div>
      <div className="memory-flow">
        {chains.map((lineage) => {
          const head = lineage[0];
          const ancestors = lineage.slice(1).reverse(); // oldest first for top-down rendering
          return (
            <div className="memory-chain" key={head.id}>
              <div style={{ fontSize: 12, color: "var(--text-faint)", marginBottom: 4 }}>
                {head.subject} · {head.predicate.replaceAll("_", " ")}
              </div>
              {ancestors.map((ancestor, i) => {
                const next = i < ancestors.length - 1 ? ancestors[i + 1] : head;
                return (
                  <div key={ancestor.id}>
                    <div
                      className={`memory-node ${ancestor.status === "ACTIVE" ? "current" : "historical"}`}
                      onClick={() => setSelected(ancestor)}
                    >
                      {ancestor.object}
                    </div>
                    <div className="memory-arrow">↓ {edgeBetween(ancestor.id, next.id)}</div>
                  </div>
                );
              })}
              <div
                className={`memory-node ${head.status === "ACTIVE" ? "current" : "historical"}`}
                onClick={() => setSelected(head)}
              >
                {head.object} {head.status === "ACTIVE" ? "· current" : `· ${head.status.toLowerCase()}`}
              </div>
            </div>
          );
        })}
      </div>

      {selected && (
        <MemoryInspector memory={selected} byId={byId} impact={impact} avgSavingsPerRun={avgSavingsPerRun} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function MemoryInspector({ memory, byId, impact, avgSavingsPerRun, onClose }) {
  const replacedBy = memory.superseded_by_memory_id ? byId[memory.superseded_by_memory_id] : null;
  const replaces = memory.supersedes_memory_id ? byId[memory.supersedes_memory_id] : null;
  const reason = memory.provenance?.extraction_reason || memory.provenance?.judgment_reason;

  return (
    <div className="memory-detail">
      <button className="btn btn-ghost" style={{ float: "right" }} onClick={onClose}>
        Close
      </button>
      <span className={`status-badge ${memory.status}`}>{memory.status}</span>
      <div style={{ fontSize: 16, fontWeight: 700, marginTop: 8 }}>{memory.content}</div>
      <dl>
        <dt>Status</dt>
        <dd>{memory.status}</dd>
        <dt>Created</dt>
        <dd>{new Date(memory.created_at).toLocaleDateString()}</dd>
        {memory.valid_until && (
          <>
            <dt>Valid until</dt>
            <dd>{new Date(memory.valid_until).toLocaleDateString()}</dd>
          </>
        )}
        <dt>Source</dt>
        <dd>Event {memory.source_event_id || "—"}</dd>
        {replaces && (
          <>
            <dt>What it replaced</dt>
            <dd>{replaces.object}</dd>
          </>
        )}
        {replacedBy && (
          <>
            <dt>What replaced it</dt>
            <dd>{replacedBy.object}</dd>
          </>
        )}
        {reason && (
          <>
            <dt>Why it exists</dt>
            <dd>{reason}</dd>
          </>
        )}
      </dl>
      {memory.status !== "ACTIVE" ? (
        <div className="impact-line">
          Cost impact:{" "}
          {impact
            ? `${formatUsd(impact, { decimals: 4 })} in AI spend avoided by keeping this out of future context.`
            : `keeping this out of future context avoids paying for it again — roughly ${
                avgSavingsPerRun ? formatUsd(avgSavingsPerRun, { decimals: 4 }) : "a small amount"
              } per question, based on recorded activity.`}
        </div>
      ) : (
        <div className="impact-line" style={{ background: "var(--blue-dim)", color: "var(--blue)" }}>
          This is the current, active memory — it's what the agent uses to answer questions about this topic today.
        </div>
      )}
    </div>
  );
}
