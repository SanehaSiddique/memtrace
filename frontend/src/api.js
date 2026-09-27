// Thin fetch wrapper. In dev, Vite proxies these paths to the FastAPI server
// (see vite.config.js); in production they're same-origin (FastAPI serves the
// built app directly), so a bare relative path works in both cases.

async function request(path, { method = "GET", body, params } = {}) {
  let url = path;
  if (params) {
    const query = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null)),
    ).toString();
    if (query) url += `?${query}`;
  }
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${method} ${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),

  listMemory: (agentId, status) => request("/memory", { params: { agent_id: agentId, status } }),
  getMemoryGraph: (memoryId) => request(`/memory/${memoryId}/graph`),
  getMemoryHistory: (memoryId) => request(`/memory/${memoryId}/history`),
  getMemoryTimeline: (agentId) => request("/memory/timeline", { params: { agent_id: agentId } }),
  getMemoryGraphFull: (agentId) => request("/memory/graph-full", { params: { agent_id: agentId } }),
  getMemoryCostImpact: (memoryId) => request(`/memory/${memoryId}/cost-impact`),

  ingest: (content, agentId, conversationId) =>
    request("/memory/ingest", { method: "POST", body: { content, agent_id: agentId, conversation_id: conversationId } }),
  chat: (message, agentId, conversationId) =>
    request("/agent/chat", { method: "POST", body: { message, agent_id: agentId, conversation_id: conversationId } }),
  query: (query, agentId, conversationId) =>
    request("/memory/query", { method: "POST", body: { query, agent_id: agentId, conversation_id: conversationId } }),
  debugQuery: (query, agentId, conversationId) =>
    request("/debug/query", { method: "POST", body: { query, agent_id: agentId, conversation_id: conversationId } }),
  debugReplay: (query, agentId, conversationId) =>
    request("/debug/replay", { method: "POST", body: { query, agent_id: agentId, conversation_id: conversationId } }),


  costSummary: (agentId) => request("/cost/summary", { params: { agent_id: agentId } }),
  costTimeseries: (agentId, granularity = "day") =>
    request("/cost/timeseries", { params: { agent_id: agentId, granularity } }),
  costLeaks: (agentId) => request("/cost/leaks", { params: { agent_id: agentId } }),
  costRuns: (agentId, limit = 20) => request("/cost/runs", { params: { agent_id: agentId, limit } }),
  memoryRoi: (agentId) => request("/cost/memory-roi", { params: { agent_id: agentId } }),
  costModels: () => request("/cost/models"),
  revenueProjection: (body) => request("/cost/revenue-projection", { method: "POST", body }),

  // Dual-Agent Benchmarking & Tracing Endpoints (docs §8)
  getGraphSnapshot: (sessionId) => request(`/api/graph/${sessionId}`),
  getSessionMetrics: (sessionId) => request(`/api/metrics/${sessionId}`),
  chatCompare: (message, sessionId = "default_session") =>
    request("/api/chat/compare", { method: "POST", body: { message, session_id: sessionId } }),
  getChatHistory: (sessionId, agentId) =>
    request(`/api/chat/${encodeURIComponent(sessionId)}/history`, { params: { agent_id: agentId } }),
  listChatSessions: () => request("/api/chat/sessions"),
  resetSession: (sessionId) =>
    request(`/api/chat/reset${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`, { method: "POST" }),

};

export const DEFAULT_AGENT_ID = "agent-alpha";
export const DEFAULT_CONVERSATION_ID = "demo_conversation";
