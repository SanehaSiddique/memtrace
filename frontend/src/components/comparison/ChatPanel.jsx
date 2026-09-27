import { useEffect, useRef, useState } from "react";
import { formatNumber, formatUsd } from "../../format";

/**
 * Reusable Chat Panel for Agent1 and Agent2 (docs/IMPLEMENTATION.md §9).
 * Subscribed to the shared WebSocket filtered by `agentId`.
 * Displays live typing status, tool call indicators, final answers, and turn metrics.
 */
export default function ChatPanel({
  agentId,
  agentTitle,
  agentSubtitle,
  tagColor = "blue",
  messages = [],
  isTyping = false,
  statusText = "",
  lastToolCall = null,
  streamingText = "",
}) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping, lastToolCall, streamingText]);

  const badgeClass = tagColor === "green" ? "badge badge-active" : "badge";

  return (
    <div className={`chat-panel card ${isTyping ? "is-live" : ""}`}>
      <div className="chat-panel-header">
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className={badgeClass}>{agentId.toUpperCase()}</span>
            <h3 style={{ margin: 0, fontSize: 16 }}>{agentTitle}</h3>
            {isTyping && (
              <span style={{ fontSize: 11, color: "var(--violet)", fontWeight: 700, display: "inline-flex", alignItems: "center" }}>
                <span className="live-dot" /> live
              </span>
            )}
          </div>
          <p className="card-note" style={{ margin: "4px 0 0 0" }}>
            {agentSubtitle}
          </p>
        </div>
      </div>

      <div className="chat-message-list" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <p>No messages yet. Send a query below to start the side-by-side benchmark.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`chat-message chat-message-${msg.role}`}>
              <div className="chat-message-header">
                <span className="chat-message-author">
                  {msg.role === "user" ? "You" : agentId === "agent1" ? "Agent 1 (Baseline)" : "Agent 2 (Neo4j + JEV)"}
                </span>
                <span className="chat-message-time">{msg.timestamp}</span>
              </div>

              <div className="chat-message-body">{msg.content}</div>

              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div className="chat-tool-calls">
                  {msg.toolCalls.map((tc, idx) => (
                    <div key={idx} className="chat-tool-pill">
                      <span className="chat-tool-icon">⚡</span>
                      <span>Tool called: <strong>{tc.tool}</strong></span>
                      {tc.latency_ms > 0 && (
                        <span className="chat-tool-latency">({Math.round(tc.latency_ms)}ms)</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {msg.metrics && (
                <div className="chat-turn-metric-footer">
                  <div className="metric-chip">
                    Tokens: <strong>{formatNumber(msg.metrics.total_context_tokens)}</strong>
                  </div>
                  {msg.metrics.raw_result_tokens != null && msg.metrics.filtered_result_tokens != null && (
                    <div className="metric-chip metric-chip-highlight">
                      Result: <strong>{msg.metrics.raw_result_tokens} → {msg.metrics.filtered_result_tokens}</strong> tok
                    </div>
                  )}
                  <div className="metric-chip">
                    Latency: <strong>{Math.round(msg.metrics.latency_ms_total)}ms</strong>
                  </div>
                  {msg.metrics.cost_projected?.["gpt-4o"] !== undefined && (
                    <div className="metric-chip">
                      Est. GPT-4o: <strong>{formatUsd(msg.metrics.cost_projected["gpt-4o"], { decimals: 4 })}</strong>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {isTyping && streamingText && (
          <div className="chat-message chat-message-assistant">
            <div className="chat-message-body">
              {streamingText}
              <span className="streaming-cursor">▊</span>
            </div>
          </div>
        )}

        {isTyping && !streamingText && (
          <div className="chat-message chat-message-assistant typing-indicator-box">
            <div className="chat-typing-dots">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
            <span style={{ fontSize: 13, color: "var(--text-dim)" }}>
              {statusText || "Processing query..."}
            </span>
          </div>
        )}

        {lastToolCall && isTyping && (
          <div className="chat-active-tool-notice">
            <span className="spinner-icon">⚙️</span>
            <span>Calling MCP tool: <code>{lastToolCall.tool}</code>...</span>
          </div>
        )}
      </div>
    </div>
  );
}
