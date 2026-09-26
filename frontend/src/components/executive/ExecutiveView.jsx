import { useCallback, useEffect, useState } from "react";
import { api, DEFAULT_AGENT_ID, DEFAULT_CONVERSATION_ID } from "../../api";
import Hero from "./Hero";
import SavingsChart from "./SavingsChart";
import MemoryRoi from "./MemoryRoi";
import ScaleCalculator from "./ScaleCalculator";
import MoneyLeaks from "./MoneyLeaks";
import BeforeAfter from "./BeforeAfter";
import WhySavedDrawer from "./WhySavedDrawer";
import MemoryEvolution from "./MemoryEvolution";
import MemoryGraph from "./MemoryGraph";
import IncidentReplay from "./IncidentReplay";
import AskBar from "./AskBar";
import RecentActivity from "./RecentActivity";
import MetricBadge from "../shared/MetricBadge";

export default function ExecutiveView({ refreshSignal }) {
  const [summary, setSummary] = useState(null);
  const [leaks, setLeaks] = useState([]);
  const [runs, setRuns] = useState([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [localSignal, setLocalSignal] = useState(0);

  const combinedSignal = refreshSignal + localSignal;

  const refresh = useCallback(() => {
    api.costSummary(DEFAULT_AGENT_ID).then(setSummary).catch(() => {});
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
          <SavingsChart agentId={DEFAULT_AGENT_ID} refreshSignal={combinedSignal} />
        </div>
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>
            Which memory decisions saved money? <MetricBadge kind="CALCULATED" />
          </h2>
          <p>Every dollar here traces back to a specific memory being kept out of the model's context.</p>
        </div>
        <MemoryRoi agentId={DEFAULT_AGENT_ID} refreshSignal={combinedSignal} />
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>Where was your AI budget going?</h2>
          <p>Based on {summary?.total_runs ?? 0} recorded run(s).</p>
        </div>
        <MoneyLeaks leaks={leaks} />
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

      <div className="section card">
        <h3 className="card-title">$0.023 looks small…</h3>
        <BeforeAfter summary={summary} />
      </div>

      <div className="section">
        <div className="section-heading">
          <h2>Memory evolution</h2>
          <p>How the agent's knowledge changed over time — click any event for detail.</p>
        </div>
        <div className="card">
          <MemoryEvolution agentId={DEFAULT_AGENT_ID} refreshSignal={combinedSignal} />
        </div>
      </div>

      <div className="section card">
        <h3 className="card-title">Your AI's memory graph</h3>
        <MemoryGraph agentId={DEFAULT_AGENT_ID} avgSavingsPerRun={summary?.avg_savings_per_run} refreshSignal={combinedSignal} />
      </div>

      <div className="section card">
        <h3 className="card-title">Find the source of the mistake</h3>
        <IncidentReplay agentId={DEFAULT_AGENT_ID} avgSavingsPerRun={summary?.avg_savings_per_run} refreshSignal={combinedSignal} />
      </div>

      <div className="section card">
        <h3 className="card-title">Recent activity</h3>
        <RecentActivity runs={runs} />
      </div>

      {showDrawer && <WhySavedDrawer summary={summary} agentId={DEFAULT_AGENT_ID} onClose={() => setShowDrawer(false)} />}
    </div>
  );
}
