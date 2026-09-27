import { useState } from "react";
import Header from "./components/shared/Header";
import ExecutiveView from "./components/executive/ExecutiveView";
import DualAgentComparisonView from "./components/comparison/DualAgentComparisonView";

export default function App() {
  const [view, setView] = useState("comparison");

  return (
    <div className="app-shell">
      <Header view={view} onChangeView={setView} />
      <div className="main-content">
        {view === "comparison" ? (
          <DualAgentComparisonView sessionId="benchmark_session_1" />
        ) : (
          <ExecutiveView />
        )}
      </div>
      <footer className="app-footer">MEMTRACE — memory and context control layer for long-running AI agents</footer>
    </div>
  );
}

