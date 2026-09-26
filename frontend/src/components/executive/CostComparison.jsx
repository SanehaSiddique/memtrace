import { formatUsd } from "../../format";

export default function CostComparison({ summary }) {
  if (!summary || summary.total_runs === 0) {
    return <div className="empty-note">No recorded runs yet — cost comparison will appear here once you ask a question.</div>;
  }
  return (
    <div className="cost-compare">
      <div className="cost-pill without">
        <div className="label">AI cost without MEMTRACE</div>
        <div className="value">{formatUsd(summary.total_cost_without_memtrace)}</div>
      </div>
      <div className="cost-arrow">→</div>
      <div className="cost-pill with">
        <div className="label">AI cost with MEMTRACE</div>
        <div className="value">{formatUsd(summary.total_cost_with_memtrace)}</div>
      </div>
      <div className="cost-pill avoided">
        <div className="label">Cost avoided</div>
        <div className="value">{formatUsd(summary.lifetime_savings)}</div>
      </div>
    </div>
  );
}
