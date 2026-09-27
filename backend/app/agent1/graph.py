"""Agent1's LangGraph flow (docs §5.3):

    START → load_stm_and_ltm → llm_reasoning_with_tools
       → [tool_call?] → call_tool (graph8, raw) → llm_final_answer → extract_facts → END
       → [no tool_call] → llm_final_answer → extract_facts → END

Everything here is the naive path, on purpose: the full tool registry goes into
every reasoning prompt, and the tool result enters the final-answer context as
its raw JSON, verbatim. The per-turn `trace` this graph returns is what the
metrics collector turns into the §7 payload.
"""

import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agent1.memory import Agent1Memory
from app.core.cost import estimate_tokens
from app.core.graph8_client import Graph8MCPClient, ToolCallResult
from app.core.llm_client import LLMClient
from app.core.system_prompt import CRM_DOMAIN_SYSTEM_PROMPT
from app.llm.errors import LLMUnavailableError
from app.metrics.schema import AgentTurnTrace
from app.tracing.langsmith import traced

AGENT_ID = "agent1"
_PROVIDER_UNAVAILABLE_ANSWER = "⚠️ The language model is temporarily unavailable (rate-limited or down). Please try again in a moment."

_AGENT1_SYSTEM_PROMPT = (
    CRM_DOMAIN_SYSTEM_PROMPT + "\n\n"
    "You have long-term memory of this session. Use the provided tools for any CRM data.\n"
    "{ltm_block}"
)

_FACT_EXTRACTION_PROMPT = (
    "What facts worth remembering did the user just state? Reply with one fact per line, "
    "no bullets, no numbering, no commentary. If the user stated no new facts, reply with "
    "the single word NONE. Store every stated fact, even if it seems to replace something "
    "the user said earlier — do not merge or compare with anything."
)


class Agent1State(TypedDict, total=False):
    session_id: str
    run_group_id: str
    user_message: str
    stm: List[dict]
    ltm_block: str
    tool_schemas: List[dict]
    pending_tool_name: Optional[str]
    pending_tool_arguments: Dict[str, Any]
    tool_result: Dict[str, Any]
    answer: str
    trace: Dict[str, Any]
    # Live "thinking process" step emitter (docs/IMPLEMENTATION_V2.md §5.2),
    # bound per-turn by orchestrator.py — Any, not a typed Callable, since it's
    # a transient runtime value that never gets serialized as part of `trace`.
    event_callback: Any


async def _emit_step(state: Agent1State, step: str, detail: Dict[str, Any]) -> None:
    callback = state.get("event_callback")
    if callback is None:
        return
    try:
        await callback(step, detail)
    except Exception:
        pass  # tracing must never break a real turn


def _serialize_tool_result(result: ToolCallResult) -> Dict[str, Any]:
    return {
        "name": result.name,
        "ok": result.ok,
        "raw_text": result.raw_text,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


def build_agent1_graph(llm: LLMClient, tools: Graph8MCPClient, memory: Agent1Memory):
    """Compile Agent1's naive flow. The shared clients are injected, never global."""

    @traced(name="agent1.load_memory")
    async def load_stm_and_ltm(state: Agent1State) -> Agent1State:
        trace = AgentTurnTrace(agent_id=AGENT_ID)
        stm = memory.history(state["session_id"])
        await _emit_step(state, "stm_loaded", {"message_count": len(stm)})
        return {
            "stm": stm,
            "ltm_block": await memory.context_block(state["session_id"]),
            "trace": trace.model_dump(),
        }

    @traced(name="agent1.reason_with_tools")
    async def llm_reasoning_with_tools(state: Agent1State) -> Agent1State:
        trace = AgentTurnTrace(**state["trace"])
        schemas = tools.tool_schemas()  # every registered schema, every turn (§5.2)
        trace.tools_considered = len(schemas)
        trace.tools_in_prompt = [s["function"]["name"] for s in schemas]

        messages: List[dict] = [
            {"role": "system", "content": _AGENT1_SYSTEM_PROMPT.format(ltm_block=state["ltm_block"])}
        ]
        messages.extend(state["stm"])
        messages.append({"role": "user", "content": state["user_message"]})

        await _emit_step(state, "llm_reasoning_start", {"tools_in_prompt": len(schemas), "model": llm.model_name})
        started = time.perf_counter()
        try:
            response = await llm.chat(
                messages,
                tools=schemas or None,
                agent_id=AGENT_ID,
                call_type="tool_selection",
                run_group_id=state["run_group_id"],
            )
        except LLMUnavailableError as exc:
            trace.tool_selection_time_ms = round((time.perf_counter() - started) * 1000, 2)
            trace.notes.append(f"llm_unavailable: {exc}")
            return {"trace": trace.model_dump(), "answer": _PROVIDER_UNAVAILABLE_ANSWER, "pending_tool_name": None}
        trace.tool_selection_time_ms = round((time.perf_counter() - started) * 1000, 2)

        if response.tool_calls:
            call = response.tool_calls[0]  # baseline: one tool per turn, no chaining
            return {
                "pending_tool_name": call.name,
                "pending_tool_arguments": call.parsed_arguments(),
                "trace": trace.model_dump(),
                "answer": "",
            }
        return {"answer": response.content, "pending_tool_name": None, "trace": trace.model_dump()}

    @traced(name="agent1.call_tool")
    async def call_tool(state: Agent1State) -> Agent1State:
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
        return {"tool_result": _serialize_tool_result(result), "trace": trace.model_dump()}

    @traced(name="agent1.final_answer")
    async def llm_final_answer(state: Agent1State) -> Agent1State:
        trace = AgentTurnTrace(**state["trace"])
        messages: List[dict] = [
            {"role": "system", "content": _AGENT1_SYSTEM_PROMPT.format(ltm_block=state["ltm_block"])}
        ]
        messages.extend(state["stm"])
        messages.append({"role": "user", "content": state["user_message"]})

        tool_result = state.get("tool_result")
        if tool_result is not None:
            # §5.2: the raw payload goes in verbatim — no filtering, no trimming.
            # Represented as plain conversational text on a "user" turn rather
            # than a native assistant.tool_calls/role:"tool" pair: this call
            # passes tools=None (no more tool use should happen), and some
            # models — Groq's gpt-oss-120b among them — keep trying to emit
            # another tool call when the history still shows one in native
            # tool-call format, which Groq then hard-rejects as "tool_choice
            # is none, but model called a tool" instead of just answering in
            # text. Ending on "user" (not a second consecutive "assistant"
            # turn) also avoids the model producing an empty completion.
            payload = tool_result["raw_text"] or f"tool error: {tool_result.get('error')}"
            messages.append(
                {
                    "role": "user",
                    "content": f"[Tool `{tool_result['name']}` result:]\n{payload}\n\nUsing this, answer my question above.",
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
            return {"answer": _PROVIDER_UNAVAILABLE_ANSWER, "trace": trace.model_dump()}
        return {"answer": response.content, "trace": trace.model_dump()}

    @traced(name="agent1.extract_facts")
    async def extract_facts(state: Agent1State) -> Agent1State:
        """§5.1: one extra LLM call, then plain inserts. No update, no dedup."""
        trace = AgentTurnTrace(**state["trace"])
        facts = await _extract_fact_lines(llm, state, trace)
        stored = await memory.remember(state["session_id"], facts)
        memory.append_turn(state["session_id"], state["user_message"], state.get("answer", ""))
        trace.details["facts_stored"] = len(stored)
        if not state.get("answer"):
            trace.notes.append("no_answer_generated")
        trace.answer = state.get("answer", "")
        return {"trace": trace.model_dump()}

    def _after_reasoning(state: Agent1State) -> str:
        if state.get("pending_tool_name"):
            return "call_tool"
        if any(note.startswith("llm_unavailable") for note in (state.get("trace") or {}).get("notes", [])):
            return "extract"  # don't fire a second doomed call at a dead provider
        return "final"

    graph = StateGraph(Agent1State)
    graph.add_node("load_stm_and_ltm", load_stm_and_ltm)
    graph.add_node("llm_reasoning_with_tools", llm_reasoning_with_tools)
    graph.add_node("call_tool", call_tool)
    graph.add_node("llm_final_answer", llm_final_answer)
    graph.add_node("extract_facts", extract_facts)

    graph.add_edge(START, "load_stm_and_ltm")
    graph.add_edge("load_stm_and_ltm", "llm_reasoning_with_tools")
    graph.add_conditional_edges("llm_reasoning_with_tools", _after_reasoning, {"call_tool": "call_tool", "final": "llm_final_answer", "extract": "extract_facts"})
    graph.add_edge("call_tool", "llm_final_answer")
    graph.add_edge("llm_final_answer", "extract_facts")
    graph.add_edge("extract_facts", END)

    return graph.compile()


async def _extract_fact_lines(llm: LLMClient, state: Agent1State, trace: AgentTurnTrace) -> List[str]:
    """Ask the model what to remember; fall back to storing the message verbatim
    (which is itself perfectly naive, and keeps the turn's memory write real
    even when the free-tier model is congested)."""
    try:
        response = await llm.chat(
            [{"role": "system", "content": _FACT_EXTRACTION_PROMPT}, {"role": "user", "content": state["user_message"]}],
            agent_id=AGENT_ID,
            call_type="fact_extraction",
            run_group_id=state["run_group_id"],
        )
    except LLMUnavailableError as exc:
        trace.notes.append(f"extraction_fallback_verbatim: {exc}")
        return _verbatim_fallback(state["user_message"])
    lines = [line.strip(" -•\t") for line in (response.content or "").splitlines()]
    facts = [line for line in lines if line and line.upper() != "NONE"]
    if not facts:
        trace.notes.append("no_facts_extracted")
    return facts


def _verbatim_fallback(user_message: str) -> List[str]:
    text = (user_message or "").strip()
    return [text] if len(text) >= 8 else []



