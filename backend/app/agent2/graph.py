"""Agent2 LangGraph Flow (docs/IMPLEMENTATION.md §6.5).

Flow:
    START → load_stm_and_ltm → jev_route_tools → llm_reasoning
       → [tool call] → call_tool (graph8) → jev_filter_result (chunk+score)
                      → llm_final_answer (filtered chunks) → extract_and_resolve_facts → END
       → [no tool]   → llm_final_answer (or reasoning answer) → extract_and_resolve_facts → END

Deltas vs Agent1:
  1. Active-facts-only from Neo4j graph LTM (no stale contradictions in prompt).
  2. JEV Choice tool routing: injects only the routed tool schema or NONE (huge token savings).
  3. JEV Score chunk filtering: prunes raw tool output down to relevant chunks before context injection.
  4. JEV Noul staleness resolution: explicitly supersedes contradicting facts in Neo4j.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agent2.chunk_filtering import filter_tool_result
from app.agent2.memory import Agent2Memory
from app.agent2.staleness import resolve_and_save_facts
from app.agent2.tool_routing import route_tools
from app.core.cost import estimate_tokens
from app.core.graph8_client import Graph8MCPClient, ToolCallResult
from app.core.jev_client import JEVClient
from app.core.llm_client import LLMClient
from app.llm.errors import LLMUnavailableError
from app.metrics.schema import AgentTurnTrace
from app.tracing.langsmith import traced

AGENT_ID = "agent2"

_AGENT2_SYSTEM_PROMPT = (
    "You are an optimized CRM assistant with graph memory and verified current facts. "
    "Answer the user's request accurately and concisely.\n"
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


def build_agent2_graph(llm: LLMClient, tools: Graph8MCPClient, jev: JEVClient, memory: Agent2Memory):
    """Compile Agent2's optimized flow with JEV routing, filtering, and staleness."""

    @traced(name="agent2.load_memory")
    async def load_stm_and_ltm(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(agent_id=AGENT_ID)
        return {
            "stm": memory.history(state["session_id"]),
            "ltm_block": await memory.context_block(state["session_id"]),
            "trace": trace.model_dump(),
        }

    @traced(name="agent2.route_tools")
    async def jev_route_tools_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
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
            return {"trace": trace.model_dump(), "answer": "", "pending_tool_name": None}

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

        return {
            "filtered_tool_text": filter_res.filtered_text,
            "trace": trace.model_dump(),
        }

    @traced(name="agent2.final_answer")
    async def llm_final_answer_node(state: Agent2State) -> Agent2State:
        trace = AgentTurnTrace(**state["trace"])
        # If reasoning already generated an answer (no tool call was made), keep it
        if state.get("answer"):
            return {"answer": state["answer"], "trace": trace.model_dump()}

        messages: List[dict] = [
            {"role": "system", "content": _AGENT2_SYSTEM_PROMPT.format(ltm_block=state["ltm_block"])}
        ]
        messages.extend(state["stm"])
        messages.append({"role": "user", "content": state["user_message"]})

        filtered_text = state.get("filtered_tool_text")
        if filtered_text is not None:
            raw_res = state.get("raw_tool_result") or {}
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": raw_res.get("name", "tool"),
                                "arguments": json.dumps(state.get("pending_tool_arguments") or {}),
                            },
                        }
                    ],
                }
            )
            # Filtered chunks, not raw bloat!
            messages.append({"role": "tool", "tool_call_id": "call_0", "content": filtered_text})

        try:
            response = await llm.chat(
                messages,
                tools=None,
                agent_id=AGENT_ID,
                call_type="final_answer",
                run_group_id=state["run_group_id"],
            )
        except LLMUnavailableError as exc:
            trace.notes.append(f"llm_unavailable_final: {exc}")
            return {"answer": "", "trace": trace.model_dump()}

        return {"answer": response.content, "trace": trace.model_dump()}

    @traced(name="agent2.extract_and_resolve_facts")
    async def extract_and_resolve_facts_node(state: Agent2State) -> Agent2State:
        """Extract facts with entities, then resolve staleness explicitly in Neo4j."""
        trace = AgentTurnTrace(**state["trace"])
        extracted = await _extract_agent2_facts(llm, state, trace)

        res = await resolve_and_save_facts(
            graph_store=memory.graph_store,
            jev=jev,
            session_id=state["session_id"],
            extracted_facts=extracted,
            run_group_id=state["run_group_id"],
        )
        memory.append_turn(state["session_id"], state["user_message"], state.get("answer", ""))

        trace.details["facts_added"] = res.new_facts_added
        trace.details["facts_superseded"] = res.facts_superseded
        trace.notes.extend(res.notes)
        if not state.get("answer"):
            trace.notes.append("no_answer_generated")
        trace.answer = state.get("answer", "")

        return {"trace": trace.model_dump()}

    def _after_reasoning(state: Agent2State) -> str:
        if state.get("pending_tool_name"):
            return "call_tool"
        if any(note.startswith("llm_unavailable") for note in (state.get("trace") or {}).get("notes", [])):
            return "extract"
        return "final"

    graph = StateGraph(Agent2State)
    graph.add_node("load_stm_and_ltm", load_stm_and_ltm)
    graph.add_node("jev_route_tools", jev_route_tools_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_node("jev_filter_result", jev_filter_result_node)
    graph.add_node("llm_final_answer", llm_final_answer_node)
    graph.add_node("extract_and_resolve_facts", extract_and_resolve_facts_node)

    graph.add_edge(START, "load_stm_and_ltm")
    graph.add_edge("load_stm_and_ltm", "jev_route_tools")
    graph.add_edge("jev_route_tools", "llm_reasoning")
    graph.add_conditional_edges(
        "llm_reasoning",
        _after_reasoning,
        {"call_tool": "call_tool", "final": "llm_final_answer", "extract": "extract_and_resolve_facts"},
    )
    graph.add_edge("call_tool", "jev_filter_result")
    graph.add_edge("jev_filter_result", "llm_final_answer")
    graph.add_edge("llm_final_answer", "extract_and_resolve_facts")
    graph.add_edge("extract_and_resolve_facts", END)

    return graph.compile()


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
