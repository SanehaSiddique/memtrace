# Agent Comparison Demo — Implementation Spec V2
### Delta from V1 — read this alongside IMPLEMENTATION.md, not instead of it

V1 sections **1, 2, 3 (repo layout — updated below), 4 (shared core layer), 5.1–5.2, 6.1–6.2, 8, 10** stay as-is unless explicitly overridden below. This document specifies six changes: codebase cleanup, a domain-constrained tool registry, async graph writes for Agent2, a live "thinking process" event stream, an updated metrics contract, and an animated clustered graph view. Everything here is written to be handed to Cline directly, same as V1.

---

## 1. Why V2

Judges need to *see* the mechanism, not just the score at the end — right now the dashboard shows outcomes (tokens, cost) but not the decision process that produced them. Separately, Agent2's first 1–2 turns were slow because graph writes were blocking the response path, and the codebase has accumulated mock/seed data from early scaffolding that needs to come out before this is demo-ready. V2 fixes all three, plus adds domain constraints so the tool-selection comparison is meaningful (choosing between 3 tools proves nothing; choosing correctly out of 50 does).

---

## 2. Codebase Cleanup (do this first, before any new feature work)

This must happen before the changes below, because new code built on top of dead/mock code inherits its problems silently.

### 2.1 Backend
Audit and remove, do not just comment out:
- Any hardcoded/seeded fact rows or Cypher `CREATE` statements used to pre-populate Postgres or Neo4j for early UI testing (search for `seed`, `sample_`, `dummy`, `mock`, `fixture` in `backend/`).
- Any stub tool implementations that returned hardcoded JSON instead of calling `graph8_client.py` for real — if `graph8_client.py` itself has a `use_mock` flag or fallback branch, remove the flag and the branch, not just default it to `False`.
- Any LLM client code path that returns a canned response instead of calling OpenRouter/Groq (e.g. a `DEV_MODE` shortcut) — remove entirely.
- Unused imports, unused Pydantic models left over from earlier schema iterations, and any `.py` file with no live import path from `main.py`'s dependency graph — verify with a simple import-trace, don't assume.
- Confirm `.env.example` only lists variables actually read by `config.py` — drop stale ones.

### 2.2 Frontend
- Remove any mock data modules (e.g. `mockMetrics.ts`, `sampleGraph.json`, hardcoded chat transcripts used to build the UI before the websocket existed).
- Remove any component that isn't reachable from `App.tsx`'s render tree — run a dead-file check (e.g. `depcheck` or manual import trace), don't eyeball it.
- Remove placeholder/lorem-ipsum content and any `TODO: replace with real data` comments — either wire the real data now or the component doesn't ship in this version.
- Confirm `package.json` dependencies match actual imports — drop unused packages (common leftover: a charting or graph library installed during exploration but replaced later).

Output of this step: a diff/PR that only removes code, changes nothing behaviorally. Verify the app still runs end-to-end against real services after cleanup before starting section 3.

---

## 3. Domain-Constrained Tool Registry

### 3.1 Rationale
Right now the tool-selection comparison is weak evidence — if there are only a handful of tools, both a naive LLM and JEV will pick correctly most of the time, and the token/latency savings from routing look marginal. Constraining both agents to a real, CRM-shaped 50-tool registry (mirroring how graph8 itself scopes agents to a domain) makes correct routing actually hard, so Agent2's JEV-routing advantage becomes measurable and visible.

### 3.2 Tool Registry Composition (50 tools total)
Curate from graph8's real MCP tool catalog into four buckets:

| Bucket | Count | Domain |
|---|---|---|
| Primary Domain A | 15 | Contacts & People (e.g. contact search, contact detail, contact history, contact notes, contact tags — exact tool names to be pulled from the live graph8 MCP tool list) |
| Primary Domain B | 15 | Companies & Deals (e.g. company search, company detail, deal/pipeline tools, account activity — exact names TBD from live catalog) |
| Auxiliary | 5 | Sequences & Campaigns (adjacent domain, still CRM-shaped, lower call frequency) |
| Mixed/Utility | 5 | Cross-cutting tools (general search, inbox lookup, or similar) that don't cleanly belong to A or B |

`[OPEN DECISION: exact tool names/counts per bucket — call graph8's MCP `list_tools` at build time and bucket the real results into this shape; the 15/15/5/5 split is a target, not a hard constraint if the real catalog doesn't divide evenly]`.

Both agents register the **identical 50-tool set** — this is required for the comparison to be fair. The difference is what each agent does with 50 tools in its prompt:
- **Agent1**: all 50 tool schemas go into every reasoning call's context, unconditionally, same as V1's design. This is now the actual stress test — 50 schemas is a meaningful fixed token cost paid every single turn regardless of relevance.
- **Agent2**: `jev_route_tool` (V1 §6.3) now runs its Choice call against all 50 tool names, and only the winning tool's schema (or zero schemas if Jev returns "none") is injected into the reasoning call.

### 3.3 System Prompt — Domain Constraint
Both agents get the same domain-scoping system prompt, so neither agent has an unfair "knows what's out of scope" advantage — the constraint is about focus/hallucination-prevention, not a routing shortcut:

```
You are a CRM assistant. You operate strictly within two domains:
1. Contacts & People — finding, viewing, and reasoning about individual contacts.
2. Companies & Deals — finding, viewing, and reasoning about companies, accounts, and deals.

You also have access to a small set of adjacent tools for sequences/campaigns and general search.

If the user asks something outside these domains, say so plainly and do not attempt to force an unrelated tool to answer it. Do not invent data — every factual claim about a contact, company, or deal must come from a tool result, never from assumption.
```

Store this as a shared constant (`backend/app/core/system_prompt.py`) imported by both `agent1/graph.py` and `agent2/graph.py` — one definition, not two copies that can drift.

---

## 4. Async Graph Writes for Agent2 (fixes early-turn latency)

### 4.1 Problem
V1's Agent2 flow (§6.5) put `extract_facts → jev_staleness_check_and_graph_write` before `END`, meaning the user's response was gated on Neo4j writes and Jev Noul contradiction-checks completing. On a cold graph (first 1–2 turns), this adds real latency and Jev-call volume to the response-critical path for no user-facing benefit — the graph update doesn't change *this* turn's answer.

### 4.2 Fix — decouple from the response path
Updated Agent2 LangGraph flow:

```
START → load_stm → jev_route_tool
   → [tool needed] → call_tool → jev_filter_result → llm_final_answer
   → [no tool] → llm_final_answer
   → RETURN ANSWER TO USER (turn ends here for the user-facing response)
   → (fire-and-forget) background_task: extract_facts → jev_staleness_check → graph_write
```

Implementation: use FastAPI `BackgroundTasks` (or `asyncio.create_task` if you need it to survive past the request/response cycle — `BackgroundTasks` runs after the response is sent but within the same request scope, which is sufficient here since the websocket connection stays open). The background task, on completion, pushes a `graph_updated` event over the same websocket connection so `GraphView.tsx` can refresh live without the user having waited for it.

### 4.3 Metrics impact
`latency_ms_total` in `TurnMetrics` now reflects only the response-critical path. Add two new fields (see §6) to track the background work separately rather than hiding it — the point of this demo is honesty about where time/tokens go, not making Agent2 look artificially fast by omission.

---

## 5. Live "Thinking Process" Visualization

### 5.1 Rationale
Judges should watch JEV make the routing decision and see the probability spread, not just read a final metrics number after the fact. This requires the backend to emit granular step-level events during a turn, not just a final response.

### 5.2 New WebSocket Event Types
Extend the `/ws/chat` event contract from V1 §8. Every step-level event carries `agent_id` and `run_group_id` (per V1 §4.4) so the frontend can route it to the correct chat panel's trace view.

```python
class TraceEvent(BaseModel):
    agent_id: str                # "agent1" | "agent2"
    run_group_id: str
    step: str                    # see step vocabulary below
    timestamp: float
    detail: dict                 # step-specific payload, see below
```

Step vocabulary and `detail` payloads:

| `step` | Emitted by | `detail` payload |
|---|---|---|
| `stm_loaded` | both | `{message_count}` |
| `jev_routing_start` | agent2 only | `{query, tool_options_count: 50}` |
| `jev_routing_result` | agent2 only | `{chosen_tool, all_probabilities: {tool_name: prob, ...}, latency_ms}` — **full probability distribution, not just the winner**, this is what makes the "how JEV guided the LLM" visual possible |
| `llm_reasoning_start` | both | `{tools_in_prompt: int, model}` |
| `tool_call_start` | both | `{tool_name}` |
| `tool_call_result` | both | `{raw_tokens_estimate}` |
| `jev_filtering_start` | agent2 only | `{chunk_count}` |
| `jev_filtering_result` | agent2 only | `{chunks: [{text_preview, score, kept: bool}], filtered_tokens_estimate}` — per-chunk scores, so the frontend can show exactly which pieces of the raw result JEV kept vs discarded |
| `llm_final_answer_start` | both | `{}` |
| `llm_final_answer_token` | both | `{token}` (streaming) |
| `turn_complete` | both | full `TurnMetrics` payload (V1 §7, extended per §6 below) |
| `graph_write_start` | agent2 only, background | `{}` |
| `graph_updated` | agent2 only, background | `{nodes_added, edges_added, stale_marked}` |

### 5.3 Frontend — `AgentTracePanel.tsx` (new component)
One instance per agent, rendered below (or beside, collapsible) each `ChatPanel.tsx`. Renders the `TraceEvent` stream for the current turn as a live, step-by-step timeline — not a log dump, a readable sequence:

- `jev_routing_result` → render as a small horizontal bar chart of `all_probabilities`, winning tool highlighted, so judges visually see "JEV considered 50 tools, scored `search_contacts` at 0.91, next-best `get_deal` at 0.34" in real time.
- `jev_filtering_result` → render each chunk with its score and a kept/discarded visual state (e.g. strikethrough or greyed for discarded), so judges see raw→filtered happening chunk by chunk.
- Agent1's trace panel is intentionally sparser (no `jev_*` events exist for it) — this asymmetry in the UI *is* part of the point; don't artificially pad Agent1's trace to make the panels look balanced.

---

## 6. Updated Metrics Schema

Extend V1 §7's `TurnMetrics`:

```python
class TurnMetrics(BaseModel):
    # ...all V1 fields unchanged...
    jev_routing_probabilities: dict[str, float] | None   # agent2 only
    jev_filtering_chunk_scores: list[float] | None        # agent2 only
    graph_write_latency_ms: float | None                  # agent2 only, reported async via graph_updated event, not part of latency_ms_total
    graph_write_jev_calls: int | None                     # staleness-check Jev calls, tracked separately from turn-critical jev_calls
```

`jev_calls` (V1 field) now counts only routing + filtering calls made during the response-critical path; staleness-check calls are counted separately in `graph_write_jev_calls` so the headline "JEV calls per turn" metric isn't inflated by background work the user never waited on.

---

## 7. Animated Clustered Graph View

### 7.1 Replace the V1 `GraphView.tsx` approach
V1 described a generic force-directed render. V2 requirement: nodes must visually cluster by `:Entity` (not float as one undifferentiated cloud), and changes must animate in rather than snap.

- Use a force-directed layout with an explicit clustering force pulling `:Fact` nodes toward their `:Entity` node's position (e.g. `d3-force`'s `forceCluster`-style custom force, or `react-force-graph`'s per-node `x`/`y` bias toward cluster centroid) — this is what makes clusters visually separable rather than one interconnected blob.
- New nodes/edges arriving via the `graph_updated` event (§4.2) must animate in (fade + settle into position), not appear instantly — this is what sells "forming in the background" to a judge watching live.
- Stale facts: visually distinct (lower opacity or desaturated color) but not removed, with the `SUPERSEDED_BY` edge rendered as a distinct style (e.g. dashed) connecting old → new — per V1 §6.1, stale facts are never deleted from the graph, only marked, so the visualization must reflect that history rather than hiding it.
- Cluster labels: render the `:Entity` name at/near each cluster's centroid so judges can read what each cluster represents without hovering every node.

---

## 8. Build Order for V2 (sequential)

1. Cleanup (§2) — ship this alone first, verify nothing broke.
2. Domain tool registry + shared system prompt (§3) — verify both agents still function correctly with 50 tools registered before touching anything else.
3. Async graph writes (§4) — verify Agent2's response latency on a cold session now matches Agent1's order of magnitude, and that `graph_updated` fires correctly after the fact.
4. Backend trace event emission (§5.2) — verify every step event fires correctly via a raw websocket client/logging before touching the frontend.
5. `AgentTracePanel.tsx` (§5.3) — wire it to the now-verified event stream.
6. Metrics schema extension (§6) — update `MetricsDashboard.tsx` to surface the new fields.
7. `GraphView.tsx` clustering/animation rework (§7) — do this last, it's the most purely visual/least functionally risky piece.
