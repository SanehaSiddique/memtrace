import { useCallback, useEffect, useState } from "react";
import { api, DEFAULT_AGENT_ID, DEFAULT_CONVERSATION_ID } from "../../api";
import Hero from "./Hero";
import CostComparison from "./CostComparison";
import SavingsChart from "./SavingsChart";
import ScaleCalculator from "./ScaleCalculator";
import MoneyLeaks from "./MoneyLeaks";
import BeforeAfter from "./BeforeAfter";
import WhySavedDrawer from "./WhySavedDrawer";
import MemoryStory from "./MemoryStory";
import IncidentReplay from "./IncidentReplay";
import AskBar from "./AskBar";
import RecentActivity from "./RecentActivity";

export default function ExecutiveView({ refreshSignal }) {
  const [summary, setSummary] = useState(null);
  const [timeseries, setTimeseries] = useState([]);
  const [leaks, setLeaks] = useState([]);
  const [runs, setRuns] = useState([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [localSignal, setLocalSignal] = useState(0);

  const refresh = useCallback(() => {
    api.costSummary(DEFAULT_AGENT_ID).then(setSummary).catch(() => {});
    api.costTimeseries(DEFAULT_AGENT_ID).then(setTimeseries).catch(() => {});
    api.costLeaks(DEFAULT_AGENT_ID).then(setLeaks).catch(() => {});
    api.costRuns(DEFAULT_AGENT_ID, 8).then(setRuns).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, refreshSignal]);

  async function handleAsk(query) {
    setBusy(true);
    try {
      const result = await api.debugQuery(query, DEFAULT_AGENT_ID, DEFAULT_CONVERSATION_ID);
      setAnswer({ query, text: result.answer });
      refresh();
      setLocalSignal((n) => n + 1);
    } finally {
      setBusy(false);
    }
  }

  async function handleTeach(message) {
    setBusy(true);
    try {
      const result = await api.chat(message, DEFAULT_AGENT_ID, DEFAULT_CONVERSATION_ID);
      setAnswer({ query: message, text: result.answer });
      refresh();
      setLocalSignal((n) => n + 1);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="section">
        <Hero summary={summary} onOpenBreakdown={() => setShowDrawer(true)} />
      </div>

      <div className="section card">
        <h3 className="card-title">AI spend, with and without MEMTRACE</h3>
        <CostComparison summary={summary} />
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>Ask your agent</h2>
          <p>Runs the full MEMTRACE pipeline and records real savings below.</p>
        </div>
        <div className="card">
          <AskBar onAsk={handleAsk} onTeach={handleTeach} busy={busy} />
          {answer && (
            <div className="answer-panel">
              <div className="q">{answer.query}</div>
              {answer.text}
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>Savings over time</h2>
          <p>Small savings per interaction become significant at scale.</p>
        </div>
        <div className="card">
          <SavingsChart points={timeseries} />
        </div>
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>What happens at your scale?</h2>
          <p>Move the sliders — this updates instantly.</p>
        </div>
        <div className="card">
          <ScaleCalculator />
        </div>
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>Where was your AI budget going?</h2>
          <p>Based on {summary?.total_runs ?? 0} recorded run(s).</p>
        </div>
        <MoneyLeaks leaks={leaks} />
      </div>

      <div className="section card">
        <h3 className="card-title">$0.023 looks small…</h3>
        <BeforeAfter summary={summary} />
      </div>

      <div className="grid grid-2 section">
        <div className="card">
          <h3 className="card-title">Your AI's memory</h3>
          <MemoryStory
            agentId={DEFAULT_AGENT_ID}
            avgSavingsPerRun={summary?.avg_savings_per_run}
            refreshSignal={refreshSignal + localSignal}
          />
        </div>
        <div className="card">
          <h3 className="card-title">Find the source of the mistake</h3>
          <IncidentReplay
            agentId={DEFAULT_AGENT_ID}
            avgSavingsPerRun={summary?.avg_savings_per_run}
            refreshSignal={refreshSignal + localSignal}
          />
        </div>
      </div>

      <div className="section card">
        <h3 className="card-title">Recent activity</h3>
        <RecentActivity runs={runs} />
      </div>

      {showDrawer && <WhySavedDrawer summary={summary} onClose={() => setShowDrawer(false)} />}
    </div>
  );
}
