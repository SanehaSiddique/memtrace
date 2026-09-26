import { useEffect, useState } from "react";
import { api } from "../../api";

function buildChains(active, historical) {
  const historicalByKey = {};
  historical.forEach((m) => {
    const key = `${m.subject}::${m.predicate}`;
    (historicalByKey[key] ||= []).push(m);
  });
  return active.map((m) => ({
    active: m,
    historical: historicalByKey[`${m.subject}::${m.predicate}`] || [],
  }));
}

export default function MemoryStory({ agentId, avgSavingsPerRun, refreshSignal }) {
  const [chains, setChains] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.listMemory(agentId, "ACTIVE"), api.listMemory(agentId, "HISTORICAL")])
      .then(([active, historical]) => setChains(buildChains(active, historical)))
      .catch(() => setChains([]))
      .finally(() => setLoading(false));
  }, [agentId, refreshSignal]);

  if (loading) return <div className="empty-note">Loading memory…</div>;
  if (chains.length === 0) {
    return <div className="empty-note">No memory yet — seed demo data to see your agent's memory here.</div>;
  }

  return (
    <div>
      <div className="memory-flow">
        {chains.map(({ active, historical }) => (
          <div className="memory-chain" key={active.id}>
            <div style={{ fontSize: 12, color: "var(--text-faint)", marginBottom: 4 }}>
              {active.subject} · {active.predicate.replaceAll("_", " ")}
            </div>
            {historical.map((h) => (
              <div key={h.id}>
                <div className="memory-node historical" onClick={() => setSelected(h)}>
                  {h.object}
                </div>
                <div className="memory-arrow">↓ superseded</div>
              </div>
            ))}
            <div className="memory-node current" onClick={() => setSelected(active)}>
              {active.object} · current
            </div>
          </div>
        ))}
      </div>

      {selected && <MemoryDetail memory={selected} avgSavingsPerRun={avgSavingsPerRun} onClose={() => setSelected(null)} />}
    </div>
  );
}

function MemoryDetail({ memory, avgSavingsPerRun, onClose }) {
  return (
    <div className="memory-detail">
      <button className="btn btn-ghost" style={{ float: "right" }} onClick={onClose}>
        Close
      </button>
      <span className={`status-badge ${memory.status}`}>{memory.status === "ACTIVE" ? "Current" : "Historical"}</span>
      <div style={{ fontSize: 16, fontWeight: 700, marginTop: 8 }}>{memory.content}</div>
      <dl>
        <dt>Status</dt>
        <dd>{memory.status}</dd>
        <dt>Created</dt>
        <dd>{new Date(memory.created_at).toLocaleDateString()}</dd>
        {memory.valid_until && (
          <>
            <dt>Superseded</dt>
            <dd>{new Date(memory.valid_until).toLocaleDateString()}</dd>
          </>
        )}
        <dt>Source</dt>
        <dd>Conversation event {memory.source_event_id || "—"}</dd>
      </dl>
      {memory.status !== "ACTIVE" && (
        <div className="impact-line">
          Cost impact: keeping this out of future context avoids paying for outdated information every time this
          topic comes up again — roughly {avgSavingsPerRun ? `$${avgSavingsPerRun.toFixed(4)}` : "a small amount"} per
          question, based on recorded activity.
        </div>
      )}
    </div>
  );
}
