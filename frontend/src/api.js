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

  seedDemo: () => request("/demo/seed", { method: "POST" }),
  listMemory: (agentId, status) => request("/memory", { params: { agent_id: agentId, status } }),
  getMemoryGraph: (memoryId) => request(`/memory/${memoryId}/graph`),
  getMemoryHistory: (memoryId) => request(`/memory/${memoryId}/history`),

  chat: (message, agentId, conversationId) =>
    request("/agent/chat", { method: "POST", body: { message, agent_id: agentId, conversation_id: conversationId } }),
  query: (query, agentId, conversationId) =>
    request("/memory/query", { method: "POST", body: { query, agent_id: agentId, conversation_id: conversationId } }),
  debugQuery: (query, agentId, conversationId) =>
    request("/debug/query", { method: "POST", body: { query, agent_id: agentId, conversation_id: conversationId } }),
  debugReplay: (query, agentId, conversationId) =>
    request("/debug/replay", { method: "POST", body: { query, agent_id: agentId, conversation_id: conversationId } }),

  evaluate: () => request("/evaluate", { method: "POST" }),

  costSummary: (agentId) => request("/cost/summary", { params: { agent_id: agentId } }),
  costTimeseries: (agentId) => request("/cost/timeseries", { params: { agent_id: agentId } }),
  costLeaks: (agentId) => request("/cost/leaks", { params: { agent_id: agentId } }),
  costRuns: (agentId, limit = 20) => request("/cost/runs", { params: { agent_id: agentId, limit } }),
  scaleProjection: (body) => request("/cost/scale-projection", { method: "POST", body }),
};

export const DEFAULT_AGENT_ID = "agent-alpha";
export const DEFAULT_CONVERSATION_ID = "demo_conversation";
