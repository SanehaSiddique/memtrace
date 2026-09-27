# Agent Comparison Demo — Implementation Spec
### Naive Baseline (Agent1) vs Graph-Memory + JEV Agent (Agent2)

This document is the single source of truth for implementation. It is written to be handed to a coding agent (Cline) directly. Every section is a contract — do not invent behavior not specified here; where a decision was deliberately left open, it is marked `[OPEN DECISION]`.

---

## 1. Goal

Run the **same user message** through two agents simultaneously and prove, with real numbers, that a graph-memory + JEV-routed agent (Agent2) uses fewer tokens, fewer LLM calls, and lower latency than a naive baseline (Agent1) — without sacrificing answer quality.

Everything must be real:
- Real LLM calls (via OpenRouter/Groq free models)
- Real DBs (Postgres for Agent1, Neo4j for Agent2)
- Real external tool (graph8 MCP server)
- Real JEV calls (TypeSafe Jev via Vercel AI Gateway)
- Real tracing (LangSmith)
- Real published pricing for the cost-projection layer (OpenAI, Anthropic, Grok)

No mocked data anywhere in the pipeline. The only thing that is "simulated" is which paid model the tokens are billed against — because the actual inference runs on free models.

---

## 2. High-Level Architecture

```
                         ┌─────────────────────────┐
                         │   React Frontend         │
                         │  - Chat Panel (Agent1)   │
                         │  - Chat Panel (Agent2)   │
                         │  - Metrics Dashboard     │
                         │  - Graph Visualization   │
                         └───────────┬─────────────┘
                                     │ WebSocket (per turn)
                         ┌───────────▼─────────────┐
                         │   FastAPI Application    │
                         │                           │
                         │  /ws/chat  (fan-out to    │
                         │   both agents per msg)    │
                         └───────────┬─────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
      ┌─────────▼────────┐ ┌────────▼─────────┐  ┌───────▼────────┐
      │ Agent1 Orchestr.  │ │ Agent2 Orchestr.  │  │  Shared Core   │
      │ (LangGraph)       │ │ (LangGraph)       │  │  Layer         │
      │ - raw tool calling│ │ - JEV tool routing│  │ - LLM client   │
      │ - raw JSON→context│ │ - JEV result filter│ │ - Jev client   │
      │ - Postgres facts  │ │ - Neo4j graph LTM │  │ - graph8 MCP   │
      └─────────┬────────┘ └────────┬─────────┘  │   client       │
                │                    │             │ - tracing hooks│
      ┌─────────▼────────┐ ┌────────▼─────────┐  │ - token/cost   │
      │  Postgres (facts) │ │  Neo4j (graph)    │  │   accounting   │
      └───────────────────┘ └───────────────────┘  └────────────────┘
```

Both orchestrators call into the **same** shared core layer (single LLM client, single graph8 MCP client, single Jev client, single tracing/cost module). Agent1 and Agent2 differ only in: (a) what they store as LTM, (b) how they decide which tool to call, (c) what they do with a tool's raw result before it re-enters context. This is deliberate — it's what makes the comparison fair and auditable.

---

## 3. Repository Layout

```
agent-comparison-demo/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, websocket endpoint
│   │   ├── config.py                # env vars, pricing tables
│   │   ├── core/
│   │   │   ├── llm_client.py        # OpenRouter/Groq wrapper, token accounting
│   │   │   ├── jev_client.py        # Vercel AI Gateway Jev wrapper (Choice/Score/Noul)
│   │   │   ├── graph8_client.py     # graph8 MCP client (shared by both agents)
│   │   │   ├── tracing.py           # LangSmith run tagging helpers
│   │   │   └── cost.py              # pricing tables + cost projection math
│   │   ├── agent1/
│   │   │   ├── graph.py             # LangGraph StateGraph for Agent1
│   │   │   ├── memory.py            # Postgres raw-fact STM/LTM read+write
│   │   │   └── tools.py             # tool schema registration (raw, undedited)
│   │   ├── agent2/
│   │   │   ├── graph.py             # LangGraph StateGraph for Agent2
│   │   │   ├── memory_graph.py      # Neo4j graph LTM read+write+staleness
│   │   │   ├── jev_routing.py       # JEV Choice call for tool selection
│   │   │   └── jev_filtering.py     # JEV Score calls for result filtering
│   │   ├── db/
│   │   │   ├── postgres.py          # SQLAlchemy models + session (Agent1)
│   │   │   └── neo4j_driver.py      # Neo4j driver + Cypher helpers (Agent2)
│   │   └── metrics/
│   │       ├── schema.py            # per-turn metrics payload (pydantic)
│   │       └── collector.py         # aggregates metrics from both agents per turn
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   │   ├── ChatPanel.tsx        # reusable, takes agentId prop
    │   │   ├── MetricsDashboard.tsx
    │   │   ├── GraphView.tsx        # Neo4j graph visualization (Agent2 tab)
    │   │   └── CostComparison.tsx   # cross-model cost bar chart
    │   ├── hooks/
    │   │   └── useAgentSocket.ts    # websocket hook, one per agent
    │   └── lib/
    │       └── metricsTypes.ts      # mirrors backend metrics/schema.py
    ├── package.json
    └── .env.example
```

---

## 4. Shared Core Layer (`backend/app/core/`)

### 4.1 `llm_client.py`
- Wraps OpenRouter and Groq behind one interface: `LLMClient.chat(messages, tools=None, model=str) -> LLMResponse`
- `LLMResponse` must carry: `content`, `tool_calls`, `usage.prompt_tokens`, `usage.completion_tokens`, `latency_ms`, `model`.
- Every call is wrapped in a LangSmith traced span (see 4.4) tagged with `agent_id` ("agent1" | "agent2") and `call_type` ("reasoning" | "tool_selection" | "final_answer").
- Model selection for the demo: read from `config.py` — default to one OpenRouter free model and one Groq free model, selectable per request for A/B robustness testing. `[OPEN DECISION: exact free model IDs — pick current OpenRouter/Groq free-tier models at build time, they rotate]`.

### 4.2 `jev_client.py`
- Wraps the Vercel AI Gateway `experimental_evaluate` call for `typesafe-ai/jev`.
- Exposes three typed methods matching Jev's primitives:
  - `jev_choice(question: str, options: list[str], context: str) -> ChoiceResult` — returns the selected option + calibrated probability per option.
  - `jev_score(question: str, context: str) -> ScoreResult` — returns a 0–1 calibrated score.
  - `jev_noul(question: str, context: str) -> NoulResult` — returns boolean + confidence.
- Every call logged with: input token estimate (Jev is billed per input token, $0 output), latency, and which of the three call-sites it came from (`tool_routing`, `result_filtering`, `staleness_check`) — this breakdown is required for the metrics dashboard.
- All Jev calls are also wrapped in a LangSmith span tagged `component="jev"`.

### 4.3 `graph8_client.py`
- Thin MCP client connecting to graph8's real MCP server (OAuth per graph8 docs, no API key needed for personal setup).
- Exposes the specific graph8 tools you decide to register for the demo — recommend starting with 2–3 read tools only (e.g. contact search, company lookup) to keep the tool-selection decision non-trivial but the result payloads bounded. Both Agent1 and Agent2 register the identical tool schemas — the only difference is what happens to the tool's raw JSON result afterward.
- **Agent1 path**: raw MCP JSON response → appended verbatim into the next LLM context turn.
- **Agent2 path**: raw MCP JSON response → passed through `jev_filtering.py` before it touches context (see 6.2).

### 4.4 `tracing.py`
- One LangSmith project per demo run (or a shared project with per-turn `trace_id` correlating both agents' runs for that turn, so metrics can be joined later). Recommended: single project `agent-comparison-demo`, every run tagged with `run_group_id` = a UUID generated per user turn, shared across Agent1's run tree and Agent2's run tree, plus `agent_id` tag to filter/split in the LangSmith UI.
- Every LLM call, every Jev call, and every tool call must be a traced run — no silent/untraced calls, or the metrics dashboard has gaps.

### 4.5 `cost.py`
- Static pricing table, one row per model: `{provider, model_name, input_price_per_1M, output_price_per_1M}`. Populate at build time with current published rates for at least: GPT-4o / GPT-4o-mini, Claude (current Sonnet/Haiku tier), Grok (current tier). `[OPEN DECISION: exact models/prices — fetch current rates at build time, these change]`.
- `project_cost(prompt_tokens, completion_tokens, pricing_row) -> float` — pure function, no LLM call. This is the "if this ran on X" calculation: take the real token counts produced by the free-model run and multiply by each paid model's published per-token price. Never invent token counts — always the actual counted usage from that turn's real LLM response.

---

## 5. Agent1 — Naive Baseline

### 5.1 Memory
- **STM**: plain chat history list (role/content), held in the LangGraph state for the session, no special treatment.
- **LTM**: Postgres table `agent1_facts`:
  ```sql
  CREATE TABLE agent1_facts (
      id SERIAL PRIMARY KEY,
      session_id TEXT NOT NULL,
      fact_text TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
  );
  ```
- On each turn: naive extraction — after the final answer, one extra LLM call asks "what facts worth remembering did the user just state?" and inserts them as new rows. **No update, no dedup, no staleness logic** — this is intentional; contradictions simply pile up as separate rows, and retrieval is "pull all rows for this session_id, dump into context." This is the flaw Agent2 exists to fix — do not accidentally make Agent1 smarter than this.
- Retrieval: every turn, all facts for `session_id` are fetched and prepended to context as plain text, unconditionally (no relevance filtering) — this is the "no fancy thing" baseline.

### 5.2 Tool calling
- Standard LLM-driven tool calling: full tool schemas (all registered tools) go into every request; the LLM decides via normal function-calling whether/which tool to invoke.
- Tool result (from `graph8_client.py`) is appended to the message history **verbatim, full raw JSON**, then a second LLM call produces the final answer.

### 5.3 LangGraph flow
```
START → load_stm_and_ltm → llm_reasoning_with_tools
   → [tool_call?] → call_tool (graph8_client, raw) → llm_final_answer → extract_facts → END
   → [no tool_call] → llm_final_answer → extract_facts → END
```

---

## 6. Agent2 — Graph Memory + JEV

### 6.1 Memory
- **STM**: identical to Agent1 — plain chat history, no special treatment. (This keeps the comparison fair: the *only* deltas are LTM structure and tool-call handling.)
- **LTM**: Neo4j graph.
  - Node label: `:Fact` with properties `{text, status: "active"|"stale", created_at, superseded_by (optional relation instead of property — see below)}`
  - Node label: `:Entity` (e.g. a topic/subject the fact is about — "programming_language_preference")
  - Relationship: `(:Fact)-[:ABOUT]->(:Entity)` — clusters facts by entity
  - Relationship: `(:Fact {status:"stale"})-[:SUPERSEDED_BY]->(:Fact {status:"active"})` — this is how staleness/history is preserved instead of deleted (never destroy a stale fact — mark it and link it, per section 6.3)
  - Session scoping: `(:Session {id})-[:HAS_FACT]->(:Fact)`

### 6.2 JEV result filtering (post-tool-call)
Design: **score-per-chunk, not whole-result classification.**

Rationale: Jev returns typed probabilistic decisions (Choice/Score/Noul), not generated text — it cannot "summarize" a JSON blob. So the contract is:

1. `graph8_client.py` returns a raw JSON tool result.
2. `jev_filtering.py` decomposes the result into a flat list of chunks — one chunk per top-level field, or per list-item if the result is a list (e.g. one chunk per contact record if the graph8 tool returned multiple contacts).
3. For each chunk, call `jev_score(question=f"Is this information relevant to answering: '{user_query}'?", context=chunk)`.
4. Keep chunks scoring above a threshold (`start at 0.5, tune empirically`), sorted descending, capped at top-N chunks (`start at N=5`) to bound worst-case Jev call volume.
5. Reassemble the surviving chunks into a compact JSON/text block — this, not the raw payload, is what enters the LLM's context for the final-answer call.
6. Batch the per-chunk Jev calls where the API supports batching (check TypeSafe's batched-call support — the reference implementation showed multi-payload batching cutting latency dramatically); if unavailable, fire them concurrently (`asyncio.gather`), not sequentially, since Jev's whole value proposition is low per-call latency that only pays off if you don't serialize it.
7. Record `raw_result_tokens` (estimate from the full raw payload) vs `filtered_result_tokens` (estimate from the surviving chunks) per turn — this is a headline metric on the dashboard.

### 6.3 JEV tool routing (pre-tool-call)
- Before any LLM reasoning call is made for a turn that might need a tool, call `jev_choice(question="Which tool, if any, is needed to answer this query?", options=[tool_names..., "none"], context=user_query + recent_stm)`.
- If Jev returns `"none"` with high confidence, skip tool-schema injection entirely for the reasoning call — the LLM call is made with zero tool definitions in the prompt, saving the fixed per-tool schema token cost on every turn (this is the "seven tools, ten tools... all of it goes into the prompt every time" problem — Jev existing specifically to kill it).
- If Jev returns a specific tool, only that tool's schema is injected — not the full registry.
- `[OPEN DECISION: confidence threshold for auto-accepting Jev's routing decision vs falling back to normal LLM-driven selection as a safety net]` — recommend starting with a hard threshold (e.g. 0.75) below which you fall back to giving the LLM the full tool list, so a bad Jev call can't silently break tool access. Log fallback frequency as a metric — it's evidence for or against JEV reliability in your writeup.

### 6.4 JEV staleness/contradiction resolution
- On fact extraction (same trigger point as Agent1 — after each turn, ask "what facts worth remembering did the user just state"), for each newly extracted candidate fact:
  1. Query Neo4j for existing active `:Fact` nodes under the same `:Entity` (or nearest entity — simple keyword/entity match is enough for the MVP, no embedding search required unless you want it).
  2. If an existing active fact is found for that entity, call `jev_noul(question="Does this new statement contradict/replace the existing fact?", context=f"existing: {old_fact}\nnew: {new_fact}")`.
  3. If Jev says yes: mark the old fact node `status="stale"`, create the new fact node `status="active"`, link `(old)-[:SUPERSEDED_BY]->(new)`.
  4. If no: both facts persist as active (they're not in conflict — e.g. "likes Python" and "has a cat" never trigger this path since they're different entities).
- This is the Java→Python example from the brief, implemented as an explicit, auditable Neo4j write, not an implicit LLM judgment call.

### 6.5 LangGraph flow
```
START → load_stm → jev_route_tool
   → [tool needed] → call_tool (graph8_client, raw)
                    → jev_filter_result (chunk+score, top-N)
                    → llm_final_answer (filtered chunks only)
   → [no tool] → llm_final_answer (no tool schemas in prompt)
   → extract_facts → jev_staleness_check_and_graph_write → END
```

---

## 7. Metrics — What Gets Tracked Per Turn, Per Agent

Define once in `backend/app/metrics/schema.py`, mirrored in `frontend/src/lib/metricsTypes.ts`:

```python
class TurnMetrics(BaseModel):
    agent_id: str                  # "agent1" | "agent2"
    run_group_id: str              # correlates both agents' runs for this turn
    tool_selection_time_ms: float
    tools_considered: int          # how many tool schemas were in the prompt
    tools_called: int
    raw_result_tokens: int | None  # None for agent1 if no tool called
    filtered_result_tokens: int | None  # None for agent1 always (no filtering step)
    total_context_tokens: int      # full prompt token count for the turn
    llm_calls: int                 # count of actual LLM calls this turn
    jev_calls: int                 # 0 for agent1 always
    latency_ms_total: float
    cost_actual_usd: float         # cost on the free model actually used (near-zero)
    cost_projected: dict[str, float]  # {"gpt-4o": x, "claude-sonnet": y, "grok": z}
```

This schema is the contract between backend and frontend — the dashboard renders directly off it, per turn, per agent, side by side.

---

## 8. FastAPI Endpoints

- `WS /ws/chat` — client sends `{message: str, session_id: str}`; server fans the message out to both Agent1's and Agent2's LangGraph app concurrently (`asyncio.gather`), streams back `{agent_id, event: "token"|"tool_call"|"final"|"metrics", data}` events as each agent progresses, so both chat panels update in parallel rather than waiting for the slower agent.
- `GET /api/graph/{session_id}` — returns the current Neo4j graph (nodes+edges, JSON) for `GraphView.tsx` to render. Poll or push via the same websocket after each `extract_facts` step in Agent2.
- `GET /api/metrics/{session_id}` — historical per-turn metrics for the session, for the dashboard's running totals/charts.

---

## 9. Frontend

- `ChatPanel.tsx` — one component, instantiated twice (`agentId="agent1"`, `agentId="agent2"`), subscribed to the shared websocket filtered by `agent_id`.
- `MetricsDashboard.tsx` — side-by-side metric cards/bars per `TurnMetrics` field, updating live per turn; running cumulative totals (total tokens saved, total LLM calls saved) across the session.
- `GraphView.tsx` — force-directed graph (e.g. `react-force-graph` or D3) rendered from `/api/graph/{session_id}`, in its own tab, updating as new nodes/edges appear — this is the "graph forming silently in the background" requirement. Stale facts should render visually distinct (e.g. greyed out) from active ones, with the `SUPERSEDED_BY` edge visible.
- `CostComparison.tsx` — bar chart of `cost_projected` per model, Agent1 vs Agent2 side by side, cumulative over the session — this is the headline "proof" visual.

---

## 10. Environment Variables

```
# backend/.env
OPENROUTER_API_KEY=
GROQ_API_KEY=
TYPESAFE_JEV_VERCEL_KEY=
GRAPH8_MCP_URL=https://be.graph8.com/mcp/
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=agent-comparison-demo
POSTGRES_URL=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
```

---

## 11. Build Order (for Cline, sequential milestones)

1. Shared core layer (`llm_client`, `cost.py`, `tracing.py`) — no agents yet, just verify a single traced LLM call works end to end with real token/cost numbers logged.
2. `graph8_client.py` — verify a real MCP tool call round-trips.
3. `jev_client.py` — verify all three primitives (Choice/Score/Noul) return against real Vercel Gateway calls.
4. Agent1 full LangGraph flow + Postgres — get one complete naive turn working, traced.
5. Agent2 full LangGraph flow + Neo4j — get one complete optimized turn working, traced, including the staleness-check path (test explicitly with a Java→Python-style contradiction).
6. Metrics collector wired to both agents' traced runs.
7. FastAPI websocket fan-out.
8. Frontend: dual chat panels first (prove parallel streaming works), then metrics dashboard, then graph view, then cost comparison chart last.

Do not build the frontend before step 4–5 produce real metrics — there is nothing honest to visualize before that.
