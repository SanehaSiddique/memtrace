import AnimatedNumber from "../shared/AnimatedNumber";
import MetricBadge from "../shared/MetricBadge";
import { formatUsd } from "../../format";

export default function Hero({ summary, onOpenBreakdown }) {
  const hasActivity = summary && summary.total_runs > 0;
  const delta = summary?.savings_change_vs_last_month ?? 0;

  return (
    <div className="hero">
      <div className="hero-eyebrow">MEMTRACE</div>
      <h1 style={{ fontSize: "clamp(22px, 3vw, 30px)", margin: "6px 0 4px 0", fontWeight: 700 }}>
        Your AI agents don't need more context.
      </h1>
      <h1 style={{ fontSize: "clamp(22px, 3vw, 30px)", margin: "0 0 6px 0", fontWeight: 700, color: "var(--text-dim)" }}>
        They need better memory.
      </h1>
      <p style={{ fontStyle: "italic", color: "var(--text-faint)", margin: "0 0 22px 0" }}>
        Memory that knows what's still true.
      </p>

      {!hasActivity && (
        <div className="hero-sub" style={{ marginBottom: 10 }}>
          No activity recorded yet. Run the live benchmark to ingest memories, then ask a question below to see savings appear.
        </div>
      )}

      <div className="grid grid-3">
        <div>
          <div className="hero-eyebrow">
            AI spend avoided <MetricBadge kind="CALCULATED" />
          </div>
          <div
            className="hero-figure"
            style={{ fontSize: "clamp(36px, 5vw, 56px)", cursor: hasActivity ? "pointer" : "default" }}
            onClick={hasActivity ? onOpenBreakdown : undefined}
            title={hasActivity ? "Click to see how this was calculated" : undefined}
          >
            <AnimatedNumber value={summary?.savings_this_month ?? 0} />
          </div>
          <div className="hero-sub">this month{hasActivity ? " — click to see how" : ""}</div>
          {summary?.savings_last_month > 0 && (
            <div className={`hero-delta ${delta >= 0 ? "up" : "down"}`}>
              {delta >= 0 ? "▲" : "▼"} {formatUsd(Math.abs(delta))} vs last month
            </div>
          )}
        </div>

        <div>
          <div className="hero-eyebrow">
            Projected annual savings <MetricBadge kind="PROJECTED" />
          </div>
          <div className="hero-figure" style={{ fontSize: "clamp(36px, 5vw, 56px)" }}>
            <AnimatedNumber value={summary?.projected_annual_savings ?? 0} />
          </div>
          <div className="hero-sub">based on current recorded usage</div>
        </div>

        <div>
          <div className="hero-eyebrow">
            Memory-driven savings <MetricBadge kind="CALCULATED" />
          </div>
          <div className="hero-figure" style={{ fontSize: "clamp(36px, 5vw, 56px)" }}>
            <AnimatedNumber value={summary?.memory_driven_savings ?? 0} />
          </div>
          <div className="hero-sub">traced to specific memory decisions</div>
        </div>
      </div>

      <div className="hero-meta-row">
        <div className="hero-meta">
          <div className="label">Agent runs protected</div>
          <div className="value">{(summary?.total_runs ?? 0).toLocaleString()}</div>
        </div>
        <div className="hero-meta">
          <div className="label">Lifetime savings</div>
          <div className="value">
            <AnimatedNumber value={summary?.lifetime_savings ?? 0} />
          </div>
        </div>
      </div>

      {hasActivity && <div className="hero-tag">{summary.data_source_note}</div>}
    </div>
  );
}
