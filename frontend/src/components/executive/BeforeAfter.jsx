import { formatUsd, formatCompactUsd } from "../../format";

export default function BeforeAfter({ summary }) {
  if (!summary || summary.total_runs === 0) {
    return <div className="empty-note">Ask a question to see a before/after cost comparison.</div>;
  }

  const avgWithout = summary.total_cost_without_memtrace / summary.total_runs;
  const avgWith = summary.total_cost_with_memtrace / summary.total_runs;
  const avgSaved = summary.avg_savings_per_run;
  const million = avgSaved * 1_000_000;

  return (
    <div>
      <div className="before-after">
        <div className="ba-col without">
          <div className="ba-title">Without MEMTRACE</div>
          <div className="ba-bars">
            <div className="ba-bar" style={{ width: "100%" }} />
            <div className="ba-bar" style={{ width: "82%" }} />
            <div className="ba-bar" style={{ width: "58%" }} />
            <div className="ba-bar" style={{ width: "34%" }} />
          </div>
          <div className="ba-cost">💸 {formatUsd(avgWithout, { decimals: 4 })} / run</div>
        </div>
        <div className="ba-col with">
          <div className="ba-title">With MEMTRACE</div>
          <div className="ba-bars">
            <div className="ba-bar" style={{ width: "42%" }} />
            <div className="ba-bar" style={{ width: "18%" }} />
          </div>
          <div className="ba-cost">💰 {formatUsd(avgWith, { decimals: 4 })} / run</div>
        </div>
      </div>

      <div className="ba-moment">
        <div className="small">{formatUsd(avgSaved, { decimals: 4 })} looks small.</div>
        <div className="small">Now multiply it by 1,000,000 runs.</div>
        <div className="big">{formatCompactUsd(million)} saved</div>
        <p className="card-note">
          Hypothetical illustration using your average recorded avoided cost per run — a projection, not a claim of
          current volume.
        </p>
      </div>
    </div>
  );
}
