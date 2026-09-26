import { formatUsd } from "../../format";

export default function WhySavedDrawer({ summary, onClose }) {
  if (!summary) return null;

  return (
    <div className="overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <button className="drawer-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <h2 style={{ marginTop: 0 }}>How MEMTRACE saved it</h2>
        <ul className="checklist">
          <li>
            <span className="check">✓</span> Removed outdated information before it reached the model
          </li>
          <li>
            <span className="check">✓</span> Avoided sending repeated or duplicate memories
          </li>
          <li>
            <span className="check">✓</span> Excluded irrelevant context
          </li>
          <li>
            <span className="check">✓</span> Retrieved only currently valid information
          </li>
          <li>
            <span className="check">✓</span> Reduced unnecessary model input, run after run
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
          <span className="k">Total avoided</span>
          <span>{formatUsd(summary.lifetime_savings)}</span>
        </div>

        <p className="card-note">{summary.data_source_note}</p>
      </div>
    </div>
  );
}
