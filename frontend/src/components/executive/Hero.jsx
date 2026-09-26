import AnimatedNumber from "../shared/AnimatedNumber";
import { formatUsd } from "../../format";

export default function Hero({ summary, onOpenBreakdown }) {
  if (!summary || summary.total_runs === 0) {
    return (
      <div className="hero">
        <div className="hero-eyebrow">Your agents saved</div>
        <div className="hero-figure">$0</div>
        <div className="hero-sub">
          No activity recorded yet. Click "Seed demo data" above, then ask a question below to see savings appear.
        </div>
      </div>
    );
  }

  const delta = summary.savings_change_vs_last_month;

  return (
    <div className="hero">
      <div className="hero-eyebrow">Your agents saved</div>
      <div className="hero-figure" onClick={onOpenBreakdown} title="Click to see how this was calculated">
        <AnimatedNumber value={summary.savings_this_month} />
      </div>
      <div className="hero-sub">this month — click the number to see how</div>

      {summary.savings_last_month > 0 && (
        <div className={`hero-delta ${delta >= 0 ? "up" : "down"}`}>
          {delta >= 0 ? "▲" : "▼"} {formatUsd(Math.abs(delta))} vs last month
        </div>
      )}

      <div className="hero-meta-row">
        <div className="hero-meta">
          <div className="label">Projected annual savings</div>
          <div className="value">
            <AnimatedNumber value={summary.projected_annual_savings} />
          </div>
        </div>
        <div className="hero-meta">
          <div className="label">Lifetime savings</div>
          <div className="value">
            <AnimatedNumber value={summary.lifetime_savings} />
          </div>
        </div>
        <div className="hero-meta">
          <div className="label">Agent runs protected</div>
          <div className="value">{summary.total_runs.toLocaleString()}</div>
        </div>
      </div>

      <div className="hero-tag">{summary.data_source_note}</div>
    </div>
  );
}
