"""Agent2 LangGraph Flow (docs/IMPLEMENTATION.md §6.5, updated by
docs/IMPLEMENTATION_V2.md §4).

Flow:
    START → load_stm_and_ltm → jev_route_tools → llm_reasoning
       → [tool call] → call_tool (graph8) → jev_filter_result (chunk+score)
                      → llm_final_answer (filtered chunks) → END (answer returned to user)
       → [no tool]   → llm_final_answer (or reasoning answer) → END (answer returned to user)
    (fire-and-forget, kicked off by orchestrator.py after the answer is sent):
       run_background_graph_write: extract_facts → jev_staleness_check → graph_write

Deltas vs Agent1:
  1. Active-facts-only from Neo4j graph LTM (no stale contradictions in prompt).
  2. JEV Choice tool routing: injects only the routed tool schema or NONE (huge token savings).
  3. JEV Score chunk filtering: prunes raw tool output down to relevant chunks before context injection.
  4. JEV Noul staleness resolution: explicitly supersedes contradicting facts in Neo4j — decoupled
     from the response path (§4.2) since a cold graph's writes/contradiction-checks added real
     latency to the response for no user-facing benefit (the graph update doesn't change this
     turn's answer). STM (`memory.append_turn`) stays on the fast path in orchestrator.py, since
     the *next* turn's history can't wait on this background work finishing.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agent2.chunk_filtering import decompose_into_chunks, filter_tool_result
from app.agent2.memory import Agent2Memory
from app.agent2.staleness import resolve_and_save_facts
from app.agent2.tool_routing import route_tools
from app.core.cost import estimate_tokens
from app.core.graph8_client import Graph8MCPClient, ToolCallResult
from app.core.jev_client import JEVClient
from app.core.llm_client import LLMClient
from app.core.system_prompt import CRM_DOMAIN_SYSTEM_PROMPT
from app.llm.errors import LLMUnavailableError
from app.metrics.schema import AgentTurnTrace
from app.tracing.langsmith import traced

AGENT_ID = "agent2"
_PROVIDER_UNAVAILABLE_ANSWER = "⚠️ The language model is temporarily unavailable (rate-limited or down). Please try again in a moment."

_AGENT2_SYSTEM_PROMPT = (
    CRM_DOMAIN_SYSTEM_PROMPT + "\n\n"
    "You have graph memory and verified current facts. Answer accurately and concisely.\n"
    "{ltm_block}"
)

_FACT_EXTRACTION_PROMPT = (
    "Extract any new facts stated by the user. For each fact, output on a separate line:\n"
    "Entity: Fact\n"
    "Example:\n"
    "Acme Corp: Switched database to PostgreSQL\n"
    "Alice: Prefers Python\n"
    "If no facts are stated, output NONE."
)


class Agent2State(TypedDict, total=False):
    session_id: str
    run_group_id: str
    user_message: str
    stm: List[dict]
    ltm_block: str
    schemas_to_inject: List[dict]
    tools_considered: int
    pending_tool_name: Optional[str]
    pending_tool_arguments: Dict[str, Any]
    raw_tool_result: Dict[str, Any]
    filtered_tool_text: str
    answer: str
    trace: Dict[str, Any]
    # Live "thinking process" step emitter (docs/IMPLEMENTATION_V2.md §5.2),
    # bound per-turn by orchestrator.py — Any, not a typed Callable, since it's
    # a transient runtime value that never gets serialized as part of `trace`.
    event_callback: Any


async def _emit_step(state: Agent2State, step: str, detail: Dict[str, Any]) -> None:
    callback = state.get("event_callback")
    if callback is None:
        return
    try:
        await callback(step, detail)
    except Exception:
        pass  # tracing must never break a real turn


@dataclass
class BackgroundGraphWriteResult:
    """Outcome of the decoupled fact-extraction + staleness-resolution + graph-write
    (§4.2) — what the orchestrator needs to build the `graph_updated` event."""

    facts_added: int = 0
    facts_superseded: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class Agent2Graph:
    """The compiled response-path graph, plus the decoupled background-write
    step (§4.2) bound to the same llm/jev/memory instances."""

    graph: Any
    run_background_graph_write: Callable[[Agent2State], Awaitable[BackgroundGraphWriteResult]]


def build_agent2_graph(llm: LLMClient, tools: Graph8MCPClient, jev: JEVClient, memory: Agent2Memory) -> Agent2Graph:
    """Compile Agent2's optimized flow with JEV routing, filtering, and staleness."""

    @traced(name="agent2.load_memory")
    async def load_stm_and_ltm(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(agent_id=AGENT_ID)
        stm = memory.history(state["session_id"])
        await _emit_step(state, "stm_loaded", {"message_count": len(stm)})
        return {
            "stm": stm,
            "ltm_block": await memory.context_block(state["session_id"]),
            "trace": trace.model_dump(),
        }

    @traced(name="agent2.route_tools")
    async def jev_route_tools_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
        all_tool_names = [s["function"]["name"] for s in tools.tool_schemas()]
        await _emit_step(
            state, "jev_routing_start", {"query": state["user_message"], "tool_options_count": len(all_tool_names)}
        )
        decision = await route_tools(
            jev=jev,
            tools=tools,
            user_message=state["user_message"],
            recent_stm=state["stm"],
            run_group_id=state["run_group_id"],
        )
        trace.tools_considered = decision.tools_considered
        trace.tool_selection_time_ms = decision.latency_ms
        trace.tools_in_prompt = [s["function"]["name"] for s in decision.schemas_to_inject]
        trace.notes.extend(decision.notes)
        trace.details["jev_routing_probabilities"] = decision.probabilities
        await _emit_step(
            state,
            "jev_routing_result",
            {
                "chosen_tool": decision.routed_choice,
                "all_probabilities": decision.probabilities,
                "latency_ms": decision.latency_ms,
            },
        )

        return {
            "schemas_to_inject": decision.schemas_to_inject,
            "tools_considered": decision.tools_considered,
            "trace": trace.model_dump(),
        }

    @traced(name="agent2.reason_with_tools")
    async def llm_reasoning_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
        schemas = state.get("schemas_to_inject") or []

        messages: List[dict] = [
            {"role": "system", "content": _AGENT2_SYSTEM_PROMPT.format(ltm_block=state["ltm_block"])}
        ]
        messages.extend(state["stm"])
        messages.append({"role": "user", "content": state["user_message"]})

        await _emit_step(state, "llm_reasoning_start", {"tools_in_prompt": len(schemas), "model": llm.model_name})
        started = time.perf_counter()
        try:
            response = await llm.chat(
                messages,
                tools=schemas if schemas else None,
                agent_id=AGENT_ID,
                call_type="tool_selection" if schemas else "reasoning",
                run_group_id=state["run_group_id"],
            )
        except LLMUnavailableError as exc:
            trace.tool_selection_time_ms += round((time.perf_counter() - started) * 1000, 2)
            trace.notes.append(f"llm_unavailable_reasoning: {exc}")
            return {"trace": trace.model_dump(), "answer": _PROVIDER_UNAVAILABLE_ANSWER, "pending_tool_name": None}

        if response.tool_calls:
            call = response.tool_calls[0]
            return {
                "pending_tool_name": call.name,
                "pending_tool_arguments": call.parsed_arguments(),
                "trace": trace.model_dump(),
                "answer": "",
            }
        return {"answer": response.content, "pending_tool_name": None, "trace": trace.model_dump()}

    @traced(name="agent2.call_tool")
    async def call_tool_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
        await _emit_step(state, "tool_call_start", {"tool_name": state.get("pending_tool_name") or ""})
        result = await tools.call_tool(
            state["pending_tool_name"] or "",
            state.get("pending_tool_arguments") or {},
            agent_id=AGENT_ID,
            run_group_id=state["run_group_id"],
        )
        trace.tools_called = 1
        trace.raw_result_tokens = estimate_tokens(result.raw_text)
        if not result.ok and result.error:
            trace.tool_errors.append(result.error)
        trace.details["tool_latency_ms"] = result.latency_ms
        trace.details["tool_name"] = result.name
        await _emit_step(state, "tool_call_result", {"raw_tokens_estimate": trace.raw_result_tokens})

        return {
            "raw_tool_result": {
                "name": result.name,
                "ok": result.ok,
                "raw_text": result.raw_text,
                "latency_ms": result.latency_ms,
                "error": result.error,
            },
            "trace": trace.model_dump(),
        }

    @traced(name="agent2.filter_tool_result")
    async def jev_filter_result_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
        raw_res = state.get("raw_tool_result") or {}
        tool_call_res = ToolCallResult(
            name=raw_res.get("name", ""),
            ok=raw_res.get("ok", False),
            raw_text=raw_res.get("raw_text", ""),
            latency_ms=raw_res.get("latency_ms", 0.0),
            error=raw_res.get("error"),
        )
        await _emit_step(state, "jev_filtering_start", {"chunk_count": len(decompose_into_chunks(tool_call_res.raw_text))})
        filter_res = await filter_tool_result(
            jev=jev,
            tool_name=tool_call_res.name,
            raw_result=tool_call_res,
            user_message=state["user_message"],
            run_group_id=state["run_group_id"],
        )
        trace.filtered_result_tokens = filter_res.filtered_tokens
        trace.notes.extend(filter_res.notes)
        trace.details["filter_chunks_total"] = filter_res.chunks_total
        trace.details["filter_chunks_kept"] = filter_res.chunks_kept
        trace.details["jev_filtering_chunk_scores"] = [
            c["score"] for c in filter_res.chunk_details if c.get("score") is not None
        ]
        await _emit_step(
            state,
            "jev_filtering_result",
            {"chunks": filter_res.chunk_details, "filtered_tokens_estimate": filter_res.filtered_tokens},
        )

        return {
            "filtered_tool_text": filter_res.filtered_text,
            "trace": trace.model_dump(),
        }

    @traced(name="agent2.final_answer")
    async def llm_final_answer_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
        # If reasoning already generated an answer (no tool call was made, or the
        # provider was unavailable during reasoning), keep it — no second doomed
        # call at a dead provider.
        if state.get("answer"):
            trace.answer = state["answer"]
            return {"answer": state["answer"], "trace": trace.model_dump()}

        messages: List[dict] = [
            {"role": "system", "content": _AGENT2_SYSTEM_PROMPT.format(ltm_block=state["ltm_block"])}
        ]
        messages.extend(state["stm"])
        messages.append({"role": "user", "content": state["user_message"]})

        filtered_text = state.get("filtered_tool_text")
        if filtered_text is not None:
            raw_res = state.get("raw_tool_result") or {}
            # Plain conversational text on a "user" turn, not a native
            # assistant.tool_calls/role:"tool" pair: this call passes
            # tools=None, and some models (Groq's gpt-oss-120b among them)
            # keep trying to emit another tool call when the history still
            # shows one in native tool-call format, which Groq then
            # hard-rejects instead of just answering in text. Ending on
            # "user" (not a second consecutive "assistant" turn) also avoids
            # the model producing an empty completion.
            messages.append(
                {
                    "role": "user",
                    # Filtered chunks, not raw bloat!
                    "content": f"[Tool `{raw_res.get('name', 'tool')}` result:]\n{filtered_text}\n\nUsing this, answer my question above.",
                }
            )

        await _emit_step(state, "llm_final_answer_start", {})

        async def _on_token(token: str) -> None:
            await _emit_step(state, "llm_final_answer_token", {"token": token})

        try:
            response = await llm.chat_stream(
                messages,
                tools=None,
                agent_id=AGENT_ID,
                call_type="final_answer",
                run_group_id=state["run_group_id"],
                on_token=_on_token,
            )
        except LLMUnavailableError as exc:
            trace.notes.append(f"llm_unavailable_final: {exc}")
            trace.answer = _PROVIDER_UNAVAILABLE_ANSWER
            return {"answer": _PROVIDER_UNAVAILABLE_ANSWER, "trace": trace.model_dump()}

        if not response.content:
            # A "successful" stream can still finish with zero content deltas
            # under provider capacity pressure (e.g. Groq cutting off a
            # reasoning model's completion once the request's prompt tokens
            # exhaust its per-minute budget) — no exception is raised, so
            # this must be checked explicitly. Never show a blank answer.
            trace.notes.append("empty_stream_completion")
            trace.answer = _PROVIDER_UNAVAILABLE_ANSWER
            return {"answer": _PROVIDER_UNAVAILABLE_ANSWER, "trace": trace.model_dump()}

        trace.answer = response.content
        return {"answer": response.content, "trace": trace.model_dump()}

    def _after_reasoning(state: Agent2State) -> str:
        # Unlike Agent1, there's no "extract" skip-branch here: llm_final_answer_node's
        # early-return (above) already passes through a reasoning-set answer (real or
        # the unavailable-fallback) without firing a second LLM call, so every path can
        # safely go through the same node.
        if state.get("pending_tool_name"):
            return "call_tool"
        return "final"

    graph = StateGraph(Agent2State)
    graph.add_node("load_stm_and_ltm", load_stm_and_ltm)
    graph.add_node("jev_route_tools", jev_route_tools_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_node("jev_filter_result", jev_filter_result_node)
    graph.add_node("llm_final_answer", llm_final_answer_node)

    graph.add_edge(START, "load_stm_and_ltm")
    graph.add_edge("load_stm_and_ltm", "jev_route_tools")
    graph.add_edge("jev_route_tools", "llm_reasoning")
    graph.add_conditional_edges(
        "llm_reasoning",
        _after_reasoning,
        {"call_tool": "call_tool", "final": "llm_final_answer"},
    )
    graph.add_edge("call_tool", "jev_filter_result")
    graph.add_edge("jev_filter_result", "llm_final_answer")
    graph.add_edge("llm_final_answer", END)

    compiled = graph.compile()

    async def _run_background_graph_write(state: Agent2State) -> BackgroundGraphWriteResult:
        """Fact extraction + JEV staleness resolution + Neo4j write, decoupled from
        the response path (§4.2). Never touches `memory.append_turn` — that's the
        caller's job on the fast path, since the next turn's STM can't wait on this."""
        trace = AgentTurnTrace(agent_id=AGENT_ID)
        extracted = await _extract_agent2_facts(llm, state, trace)
        res = await resolve_and_save_facts(
            graph_store=memory.graph_store,
            jev=jev,
            session_id=state["session_id"],
            extracted_facts=extracted,
            run_group_id=state.get("run_group_id", ""),
        )
        return BackgroundGraphWriteResult(
            facts_added=res.new_facts_added,
            facts_superseded=res.facts_superseded,
            notes=list(trace.notes) + list(res.notes),
        )

    return Agent2Graph(graph=compiled, run_background_graph_write=_run_background_graph_write)


async def _extract_agent2_facts(llm: LLMClient, state: Agent2State, trace: AgentTurnTrace) -> List[Tuple[str, str]]:
    """Extract (fact, entity) pairs from user message."""
    try:
        response = await llm.chat(
            [{"role": "system", "content": _FACT_EXTRACTION_PROMPT}, {"role": "user", "content": state["user_message"]}],
            agent_id=AGENT_ID,
            call_type="fact_extraction",
            run_group_id=state["run_group_id"],
        )
    except LLMUnavailableError as exc:
        trace.notes.append(f"extraction_fallback_verbatim: {exc}")
        text = (state["user_message"] or "").strip()
        return [(text, "General")] if len(text) >= 8 else []

    pairs: List[Tuple[str, str]] = []
    lines = [line.strip() for line in (response.content or "").splitlines() if line.strip()]
    for line in lines:
        if line.upper() == "NONE":
            continue
        if ":" in line:
            parts = line.split(":", 1)
            entity = parts[0].strip(" -*")
            fact = parts[1].strip()
            if entity and fact:
                pairs.append((fact, entity))
        elif len(line) >= 8:
            pairs.append((line, "General"))

    if not pairs:
        trace.notes.append("no_facts_extracted")
    return pairs
