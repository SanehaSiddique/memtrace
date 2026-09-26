import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";

const STATUS_DOT = {
  ACTIVE: "var(--green)",
  HISTORICAL: "var(--amber)",
  PENDING_REVIEW: "var(--purple)",
  ARCHIVED: "var(--text-faint)",
  DELETED: "var(--text-faint)",
};

function operationFor(memory) {
  return memory.supersedes_memory_id ? "UPDATE" : "ADD";
}

export default function MemoryEvolution({ agentId, refreshSignal }) {
  const [timeline, setTimeline] = useState([]);
  const [byId, setById] = useState({});
  const [selectedId, setSelectedId] = useState(null);
  const [impact, setImpact] = useState(null);

  useEffect(() => {
    api
      .getMemoryTimeline(agentId)
      .then((list) => {
        setTimeline(list);
        setById(Object.fromEntries(list.map((m) => [m.id, m])));
      })
      .catch(() => {});
  }, [agentId, refreshSignal]);

  useEffect(() => {
    if (!selectedId) {
      setImpact(null);
      return;
    }
    api
      .getMemoryCostImpact(selectedId)
      .then((r) => setImpact(r.cost_avoided))
      .catch(() => setImpact(null));
  }, [selectedId]);

  if (timeline.length === 0) {
    return (
      <div className="empty-note">No memory events yet — seed demo data to see how the agent's knowledge evolved.</div>
    );
  }

  const selected = selectedId ? byId[selectedId] : null;

  return (
    <div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {timeline.map((m, i) => (
          <div key={m.id} style={{ display: "flex", gap: 14 }}>
            <div style={{ width: 76, flexShrink: 0, color: "var(--text-faint)", fontSize: 12, paddingTop: 9, textAlign: "right" }}>
              {new Date(m.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
            </div>
            <div
              style={{
                flex: 1,
                borderLeft: "2px solid var(--border)",
                paddingLeft: 18,
                paddingBottom: i < timeline.length - 1 ? 18 : 2,
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: -7,
                  top: 5,
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  background: STATUS_DOT[m.status] || "var(--text-faint)",
                  border: "2px solid var(--bg)",
                }}
              />
              <div className="memory-node" style={{ display: "inline-block" }} onClick={() => setSelectedId(m.id)}>
                <span className={`badge ${m.status}`}>{m.status}</span>
                {m.subject} {m.predicate.replaceAll("_", " ")} {m.object}
                {operationFor(m) === "UPDATE" && byId[m.supersedes_memory_id] && (
                  <span style={{ marginLeft: 8, color: "var(--text-faint)", fontSize: 12 }}>
                    · replaced {byId[m.supersedes_memory_id].object}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <div className="memory-detail" style={{ marginTop: 20 }}>
          <button className="btn btn-ghost" style={{ float: "right" }} onClick={() => setSelectedId(null)}>
            Close
          </button>
          <span className="status-badge ACTIVE" style={{ background: "var(--blue-dim)", color: "var(--blue)" }}>
            {operationFor(selected)}
          </span>
          <div style={{ fontSize: 16, fontWeight: 700, marginTop: 8 }}>"{selected.content}"</div>
          <dl>
            <dt>Operation</dt>
            <dd>{operationFor(selected)}</dd>
            {selected.supersedes_memory_id && byId[selected.supersedes_memory_id] && (
              <>
                <dt>Old</dt>
                <dd>{byId[selected.supersedes_memory_id].object} → HISTORICAL</dd>
              </>
            )}
            <dt>New</dt>
            <dd>
              {selected.object} → {selected.status}
            </dd>
            <dt>Timestamp</dt>
            <dd>{new Date(selected.created_at).toLocaleString()}</dd>
            <dt>Source</dt>
            <dd>Event {selected.source_event_id || "—"}</dd>
            <dt>Why</dt>
            <dd>{selected.provenance?.extraction_reason || selected.provenance?.judgment_reason || "—"}</dd>
          </dl>
          {impact !== null && impact > 0 && (
            <div className="impact-line">
              Cost impact: {formatUsd(impact, { decimals: 4 })} estimated AI spend avoided so far by this memory
              decision.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
