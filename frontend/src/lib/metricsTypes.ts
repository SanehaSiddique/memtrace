/**
 * Contract between backend (backend/app/metrics/schema.py) and frontend.
 * Matches docs/IMPLEMENTATION.md §7.
 */

export interface TurnMetrics {
  agent_id: "agent1" | "agent2";
  run_group_id: string;
  tool_selection_time_ms: number;
  tools_considered: number;
  tools_called: number;
  raw_result_tokens: number | null;
  filtered_result_tokens: number | null;
  total_context_tokens: number;
  completion_tokens?: number;
  llm_calls: number;
  llm_provider_calls?: number;
  llm_cache_hits?: number;
  jev_calls: number;
  jev_breakdown?: Record<string, number>;
  jev_errors?: number;
  latency_ms_total: number;
  cost_actual_usd: number;
  cost_projected: Record<string, number>; // { "gpt-4o": x, "claude-sonnet": y, "grok": z }

  session_id?: string;
  turn_index?: number;
  user_message?: string;
  answer?: string;
  model?: string;
  tool_names?: string[];
  tool_errors?: string[];
  notes?: string[];
  store_backends?: Record<string, string>;
  created_at?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  toolCalls?: Array<{ tool: string; latency_ms?: number }>;
  metrics?: TurnMetrics;
  status?: string;
  timestamp: string;
}

export interface GraphNode {
  id: string;
  text: string;
  status: "ACTIVE" | "STALE" | "REPLACED";
  created_at?: string;
  turn_index?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string; // "SUPERSEDED_BY" etc.
}

export interface GraphSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface SessionSummary {
  session_id: string;
  total_turns: number;
  agent1: {
    turns_count: number;
    total_context_tokens: number;
    total_llm_calls: number;
    projected_cost_gpt4o: number;
    projected_cost_claude: number;
  };
  agent2: {
    turns_count: number;
    total_context_tokens: number;
    total_llm_calls: number;
    total_jev_calls: number;
    projected_cost_gpt4o: number;
    projected_cost_claude: number;
  };
  savings: {
    tokens_saved: number;
    tokens_saved_pct: number;
    llm_calls_saved: number;
    cost_saved_gpt4o: number;
    cost_saved_claude: number;
  };
}

export interface WebSocketEvent {
  agent_id: "agent1" | "agent2" | "all";
  event: "token" | "tool_call" | "final" | "metrics" | "graph" | "summary" | "status" | "error";
  data: any;
  run_group_id: string;
}
