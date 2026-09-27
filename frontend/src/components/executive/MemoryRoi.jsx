import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";
import MetricBadge from "../shared/MetricBadge";

export default function MemoryRoi({ agentId, refreshSignal }) {
  const [roi, setRoi] = useState(null);

  useEffect(() => {
    api.memoryRoi(agentId).then(setRoi).catch(() => {});
  }, [agentId, refreshSignal]);

  if (!roi) return <div className="empty-note">Loading…</div>;

  const hasActivity = roi.rows.some((r) => r.event_count > 0);

  return (
    <div className="card">
      <h3 className="card-title">
        Memory ROI <MetricBadge kind="CALCULATED" />
      </h3>
      {!hasActivity ? (
        <div className="empty-note">
          No memory decisions recorded yet — ask a few questions to see which decisions are saving
          money.
        </div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--text-dim)", fontSize: 11, textTransform: "uppercase" }}>
              <th style={{ padding: "8px 0" }}>Memory action</th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>Events</th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>Cost avoided</th>
            </tr>
          </thead>
          <tbody>
            {roi.rows.map((row) => (
              <tr key={row.operation} style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "10px 0" }}>{row.label}</td>
                <td style={{ padding: "10px 0", textAlign: "right", color: "var(--text-dim)" }}>{row.event_count}</td>
                <td style={{ padding: "10px 0", textAlign: "right", fontWeight: 700 }}>{formatUsd(row.cost_avoided)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: "2px solid var(--border)" }}>
              <td style={{ padding: "12px 0", fontWeight: 800, textTransform: "uppercase", fontSize: 12 }}>
                Total memory-driven savings
              </td>
              <td></td>
              <td style={{ padding: "12px 0", textAlign: "right", fontWeight: 800, fontSize: 18, color: "var(--green)" }}>
                {formatUsd(roi.total_cost_avoided)}
              </td>
            </tr>
          </tfoot>
        </table>
      )}
      <p className="card-note">{roi.data_source_note}</p>
    </div>
  );
}
