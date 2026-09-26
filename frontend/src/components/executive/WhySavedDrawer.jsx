import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";

export default function WhySavedDrawer({ summary, agentId, onClose }) {
  const [roi, setRoi] = useState(null);

  useEffect(() => {
    api.memoryRoi(agentId).then(setRoi).catch(() => {});
  }, [agentId]);

  if (!summary) return null;

  const countFor = (op) => roi?.rows.find((r) => r.operation === op)?.event_count ?? 0;

  return (
    <div className="overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <button className="drawer-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <h2 style={{ marginTop: 0 }}>Why did MEMTRACE save money?</h2>
        <ul className="checklist">
          <li>
            <span className="check">✓</span> {countFor("outdated_information").toLocaleString()} stale memories
            excluded
          </li>
          <li>
            <span className="check">✓</span> {countFor("deduplicated_memory").toLocaleString()} duplicate memories
            avoided
          </li>
          <li>
            <span className="check">✓</span> {countFor("irrelevant_context").toLocaleString()} irrelevant memories
            excluded
          </li>
          <li>
            <span className="check">✓</span> {countFor("unverified_information").toLocaleString()} unverified
            memories held back
          </li>
          <li>
            <span className="check">✓</span> {countFor("archived_obsolete").toLocaleString()} obsolete memories
            archived
          </li>
        </ul>

        <div className="kv-row">
          <span className="k">Across</span>
          <span>{summary.total_runs.toLocaleString()} agent runs</span>
        </div>
        <div className="kv-row">
          <span className="k">Average avoided cost per run</span>
          <span>{formatUsd(summary.avg_savings_per_run, { decimals: 4 })}</span>
        </div>
        <div className="kv-row">
          <span className="k">Estimated AI spend avoided</span>
          <span style={{ fontWeight: 800, color: "var(--green)" }}>{formatUsd(summary.lifetime_savings)}</span>
        </div>

        <p className="card-note">{summary.data_source_note}</p>
      </div>
    </div>
  );
}
