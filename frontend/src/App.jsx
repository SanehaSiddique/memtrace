import { useState } from "react";
import Header from "./components/shared/Header";
import ExecutiveView from "./components/executive/ExecutiveView";
import EngineeringView from "./components/engineering/EngineeringView";
import DualAgentComparisonView from "./components/comparison/DualAgentComparisonView";
import { api } from "./api";

export default function App() {
  const [view, setView] = useState("comparison");
  const [seeding, setSeeding] = useState(false);
  const [refreshSignal, setRefreshSignal] = useState(0);

  async function handleSeed() {
    setSeeding(true);
    try {
      await api.seedDemo();
      setRefreshSignal((n) => n + 1);
    } finally {
      setSeeding(false);
    }
  }

  return (
    <div className="app-shell">
      <Header view={view} onChangeView={setView} onSeed={handleSeed} seeding={seeding} />
      <div className="main-content">
        {view === "comparison" ? (
          <DualAgentComparisonView sessionId="benchmark_session_1" />
        ) : view === "executive" ? (
          <ExecutiveView refreshSignal={refreshSignal} />
        ) : (
          <EngineeringView refreshSignal={refreshSignal} />
        )}
      </div>
      <footer className="app-footer">MEMTRACE — memory and context control layer for long-running AI agents</footer>
    </div>
  );
}

