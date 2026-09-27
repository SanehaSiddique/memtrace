import { useEffect, useState, useCallback, useRef } from "react";
import ChatPanel from "./ChatPanel";
import AgentTracePanel from "./AgentTracePanel";
import MetricsDashboard from "./MetricsDashboard";
import GraphView from "./GraphView";
import CostComparison from "./CostComparison";
import { useAgentSocket } from "../../hooks/useAgentSocket";
import { api } from "../../api";

const BENCHMARK_PROMPTS = [
  "Find contact details for Acme Corp lead",
  "We decided yesterday to migrate our backend from MySQL to PostgreSQL",
  "What database are we using right now?",
  "List our current top enterprise partners",
];

export default function DualAgentComparisonView({ sessionId = "default_session" }) {
  const [inputText, setInputText] = useState("");
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "metrics" | "graph" | "cost"

  // Messages per agent
  const [a1Messages, setA1Messages] = useState([]);
  const [a2Messages, setA2Messages] = useState([]);

  // Live status & tool calls
  const [a1Typing, setA1Typing] = useState(false);
  const [a2Typing, setA2Typing] = useState(false);
  const [a1Status, setA1Status] = useState("");
  const [a2Status, setA2Status] = useState("");
  const [a1LastTool, setA1LastTool] = useState(null);
  const [a2LastTool, setA2LastTool] = useState(null);

  // Live "thinking process" trace for the current turn (docs/IMPLEMENTATION_V2.md §5.3)
  const [a1Trace, setA1Trace] = useState([]);
  const [a2Trace, setA2Trace] = useState([]);
  const [a1StreamingText, setA1StreamingText] = useState("");
  const [a2StreamingText, setA2StreamingText] = useState("");

  // Latest turn metrics & summary
  const [latestA1Metrics, setLatestA1Metrics] = useState(null);
  const [latestA2Metrics, setLatestA2Metrics] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [graphSignal, setGraphSignal] = useState(0);

  const { connected, connecting, sendMessage, subscribe } = useAgentSocket({ sessionId });

  // Initial load of historical metrics
  useEffect(() => {
    api
      .getSessionMetrics(sessionId)
      .then((data) => {
        if (data && data.summary) {
          setSessionSummary(data.summary);
        }
      })
      .catch(() => {});
  }, [sessionId]);

  // Handle incoming WebSocket events
  useEffect(() => {
    const unsubscribe = subscribe((event) => {
      const now = new Date().toLocaleTimeString();

      // TraceEvents (docs/IMPLEMENTATION_V2.md §5.2) carry `step`+`timestamp`+`detail`
      // instead of ChatEvent's `event`+`data` — distinguish by shape, same connection.
      if (event.step) {
        if (event.agent_id === "agent1") {
          setA1Trace((prev) => [...prev, event]);
          if (event.step === "llm_final_answer_token") {
            setA1StreamingText((prev) => prev + (event.detail?.token || ""));
          }
        } else if (event.agent_id === "agent2") {
          setA2Trace((prev) => [...prev, event]);
          if (event.step === "llm_final_answer_token") {
            setA2StreamingText((prev) => prev + (event.detail?.token || ""));
          }
        }
        return;
      }

      if (event.event === "status") {
        if (event.agent_id === "agent1") {
          setA1Typing(true);
          setA1Status(event.data?.status || "Reasoning...");
          setA1Trace([]);
          setA1StreamingText("");
        } else if (event.agent_id === "agent2") {
          setA2Typing(true);
          setA2Status(event.data?.status || "Reasoning...");
          setA2Trace([]);
          setA2StreamingText("");
        }
      } else if (event.event === "tool_call") {
        if (event.agent_id === "agent1") {
          setA1LastTool(event.data);
          setA1Status(`Tool executed: ${event.data.tool}`);
        } else if (event.agent_id === "agent2") {
          setA2LastTool(event.data);
          setA2Status(`Tool executed: ${event.data.tool}`);
        }
      } else if (event.event === "final") {
        const text = event.data?.answer || "";
        if (event.agent_id === "agent1") {
          setA1Messages((prev) => [
            ...prev,
            {
              id: `a1_${Date.now()}_${Math.random()}`,
              role: "assistant",
              content: text,
              toolCalls: a1LastTool ? [a1LastTool] : [],
              timestamp: now,
            },
          ]);
          setA1Typing(false);
          setA1LastTool(null);
          setA1StreamingText("");
        } else if (event.agent_id === "agent2") {
          setA2Messages((prev) => [
            ...prev,
            {
              id: `a2_${Date.now()}_${Math.random()}`,
              role: "assistant",
              content: text,
              toolCalls: a2LastTool ? [a2LastTool] : [],
              timestamp: now,
            },
          ]);
          setA2Typing(false);
          setA2LastTool(null);
          setA2StreamingText("");
        }
      } else if (event.event === "metrics") {
        if (event.agent_id === "agent1") {
          setLatestA1Metrics(event.data);
          // Attach metrics to latest assistant message
          setA1Messages((prev) => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (updated[lastIdx].role === "assistant") {
              updated[lastIdx] = { ...updated[lastIdx], metrics: event.data };
            }
            return updated;
          });
        } else if (event.agent_id === "agent2") {
          setLatestA2Metrics(event.data);
          setA2Messages((prev) => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (updated[lastIdx].role === "assistant") {
              updated[lastIdx] = { ...updated[lastIdx], metrics: event.data };
            }
            return updated;
          });
        }
      } else if (event.event === "graph" || event.event === "graph_updated") {
        // "graph_updated" fires once agent2's decoupled fact-extraction/staleness/
        // write finishes in the background (docs/IMPLEMENTATION_V2.md §4.2) — same
        // refetch-trigger as "graph", just arriving later, after the write lands.
        setGraphSignal((n) => n + 1);
      } else if (event.event === "summary") {
        setSessionSummary(event.data);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [subscribe, a1LastTool, a2LastTool]);

  const handleSend = () => {
    const text = inputText.trim();
    if (!text) return;

    const now = new Date().toLocaleTimeString();
    const userMsg = {
      id: `usr_${Date.now()}_${Math.random()}`,
      role: "user",
      content: text,
      timestamp: now,
    };

    // Both chat panels show the user query
    setA1Messages((prev) => [...prev, userMsg]);
    setA2Messages((prev) => [...prev, userMsg]);

    setA1Typing(true);
    setA2Typing(true);
    setA1Status("Waiting for response...");
    setA2Status("Waiting for response...");

    sendMessage(text);
    setInputText("");
  };

  const handleReset = async () => {
    try {
      await api.resetSession(sessionId);
      setA1Messages([]);
      setA2Messages([]);
      setLatestA1Metrics(null);
      setLatestA2Metrics(null);
      setSessionSummary(null);
      setGraphSignal((n) => n + 1);
      setA1Trace([]);
      setA2Trace([]);
      setA1StreamingText("");
      setA2StreamingText("");
    } catch (e) {
      console.warn("Reset failed:", e);
    }
  };

  return (
    <div className="dual-comparison-root">
      {/* Top Banner / Tab Navigation */}
      <div className="comparison-tabs-header card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", gap: 4, background: "var(--panel-2)", padding: 4, borderRadius: 999, border: "1px solid var(--border)" }}>
              <button
                className={`btn btn-ghost ${activeTab === "chat" ? "active-tab" : ""}`}
                style={{ padding: "6px 14px", borderRadius: 999, fontSize: 13, background: activeTab === "chat" ? "var(--text)" : "transparent", color: activeTab === "chat" ? "var(--bg)" : "var(--text)" }}
                onClick={() => setActiveTab("chat")}
              >
                Side-by-Side Chat
              </button>
              <button
                className={`btn btn-ghost ${activeTab === "metrics" ? "active-tab" : ""}`}
                style={{ padding: "6px 14px", borderRadius: 999, fontSize: 13, background: activeTab === "metrics" ? "var(--text)" : "transparent", color: activeTab === "metrics" ? "var(--bg)" : "var(--text)" }}
                onClick={() => setActiveTab("metrics")}
              >
                Metrics Dashboard
              </button>
              <button
                className={`btn btn-ghost ${activeTab === "graph" ? "active-tab" : ""}`}
                style={{ padding: "6px 14px", borderRadius: 999, fontSize: 13, background: activeTab === "graph" ? "var(--text)" : "transparent", color: activeTab === "graph" ? "var(--bg)" : "var(--text)" }}
                onClick={() => setActiveTab("graph")}
              >
                Neo4j Graph View
              </button>
              <button
                className={`btn btn-ghost ${activeTab === "cost" ? "active-tab" : ""}`}
                style={{ padding: "6px 14px", borderRadius: 999, fontSize: 13, background: activeTab === "cost" ? "var(--text)" : "transparent", color: activeTab === "cost" ? "var(--bg)" : "var(--text)" }}
                onClick={() => setActiveTab("cost")}
              >
                Cost Comparison
              </button>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 12, color: connected ? "var(--green)" : connecting ? "var(--amber)" : "var(--red)" }}>
              ● {connected ? "Live WebSocket Connected" : connecting ? "Connecting..." : "HTTP Fallback Mode"}
            </span>
            <button className="btn btn-ghost" onClick={handleReset} style={{ fontSize: 12 }}>
              Reset Session
            </button>
          </div>
        </div>
      </div>

      {/* Main Tab Content */}
      {activeTab === "chat" && (
        <div className="tab-pane-chat">
          <div className="grid grid-2" style={{ marginBottom: 20 }}>
            <div>
              <ChatPanel
                agentId="agent1"
                agentTitle="Agent 1 (Naive Baseline)"
                agentSubtitle="Flat Postgres facts (active + stale) + all tools loaded + raw unfiltered tool payloads"
                tagColor="red"
                messages={a1Messages}
                isTyping={a1Typing}
                statusText={a1Status}
                lastToolCall={a1LastTool}
                streamingText={a1StreamingText}
              />
              <AgentTracePanel agentId="agent1" events={a1Trace} />
            </div>
            <div>
              <ChatPanel
                agentId="agent2"
                agentTitle="Agent 2 (Neo4j + JEV Memory Layer)"
                agentSubtitle="Active Neo4j facts + JEV tool routing + JEV score chunk filtering + JEV noul staleness"
                tagColor="green"
                messages={a2Messages}
                isTyping={a2Typing}
                statusText={a2Status}
                lastToolCall={a2LastTool}
                streamingText={a2StreamingText}
              />
              <AgentTracePanel agentId="agent2" events={a2Trace} />
            </div>
          </div>

          {/* Quick Benchmark Prompt Suggestions */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "var(--text-dim)", fontWeight: 600 }}>Demo Benchmark Prompts:</span>
              {BENCHMARK_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  className="btn btn-ghost"
                  style={{ fontSize: 12, border: "1px solid var(--border)", padding: "4px 10px" }}
                  onClick={() => {
                    setInputText(prompt);
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* User Input Bar */}
          <div className="card ask-bar-container">
            <div className="ask-bar" style={{ display: "flex", gap: 10 }}>
              <input
                type="text"
                placeholder="Ask both agents concurrently (e.g. 'What database are we using now?' or 'Find contact for Acme')..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                style={{ flex: 1, padding: "12px 16px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--panel-2)", color: "var(--text)", fontSize: 14 }}
              />
              <button
                className="btn btn-primary"
                onClick={handleSend}
                disabled={!inputText.trim() || (a1Typing && a2Typing)}
                style={{ minWidth: 120, fontSize: 14 }}
              >
                {a1Typing || a2Typing ? "Running..." : "Send Prompt"}
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === "metrics" && (
        <MetricsDashboard
          latestA1Metrics={latestA1Metrics}
          latestA2Metrics={latestA2Metrics}
          sessionSummary={sessionSummary}
        />
      )}

      {activeTab === "graph" && (
        <GraphView sessionId={sessionId} refreshSignal={graphSignal} />
      )}

      {activeTab === "cost" && (
        <CostComparison
          sessionSummary={sessionSummary}
          latestA1Metrics={latestA1Metrics}
          latestA2Metrics={latestA2Metrics}
        />
      )}
    </div>
  );
}
