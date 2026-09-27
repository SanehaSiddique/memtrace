"""Dual-Agent Comparison Orchestrator (docs/IMPLEMENTATION.md §8).

Runs Agent1 (naive baseline) and Agent2 (Neo4j + JEV) concurrently via
`asyncio.gather`, streaming real-time events, tool calls, final answers, and
turn metrics side-by-side to the frontend over WebSocket.
"""

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from app.agent1.graph import build_agent1_graph
from app.agent1.memory import Agent1Memory
from app.agent2.graph import Agent2State, BackgroundGraphWriteResult, build_agent2_graph
from app.agent2.memory import Agent2Memory
from app.core.graph8_client import Graph8MCPClient
from app.core.jev_client import JEVClient
from app.core.llm_client import LLMClient
from app.metrics.collector import SessionMetricsStore, build_turn_metrics
from app.metrics.schema import AgentTurnTrace, ChatEvent, TraceEvent, TurnMetrics

EventCallback = Callable[[ChatEvent], Awaitable[None]]
# What a graph node calls to emit one live "thinking process" step (docs/IMPLEMENTATION_V2.md
# §5.2) — (step, detail); the orchestrator (which knows agent_id/run_group_id) wraps it into
# a TraceEvent and sends it over the same websocket as the turn-level ChatEvents.
StepEmitter = Callable[[str, Dict[str, Any]], Awaitable[None]]


class ComparisonOrchestrator:
    """Manages concurrent dual-agent execution and event broadcasting."""

    def __init__(
        self,
        llm: LLMClient,
        tools: Graph8MCPClient,
        jev: JEVClient,
        agent1_memory: Agent1Memory,
        agent2_memory: Agent2Memory,
        metrics_store: Optional[SessionMetricsStore] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.jev = jev
        self.agent1_memory = agent1_memory
        self.agent2_memory = agent2_memory
        self.metrics_store = metrics_store or SessionMetricsStore()

        self.agent1_graph = build_agent1_graph(llm=llm, tools=tools, memory=agent1_memory)
        self.agent2_graph = build_agent2_graph(llm=llm, tools=tools, jev=jev, memory=agent2_memory)
        # Strong references to in-flight background graph-writes (docs/IMPLEMENTATION_V2.md
        # §4.2) — asyncio.create_task's result must be held onto, or the task can be
        # garbage-collected mid-run.
        self._background_tasks: set = set()

    async def run_comparison_turn(
        self,
        session_id: str,
        user_message: str,
        event_callback: Optional[EventCallback] = None,
    ) -> Dict[str, Any]:
        """Execute Agent1 and Agent2 side-by-side on the same user prompt (§8)."""
        run_group_id = f"turn_{uuid.uuid4().hex[:10]}"
        turn_index = self.metrics_store.get_turn_count(session_id) + 1

        async def _emit(event: ChatEvent) -> None:
            if event_callback is not None:
                try:
                    await event_callback(event)
                except Exception:
                    pass  # never crash runner on websocket write error

        async def _emit_trace(agent_id: str, step: str, detail: Dict[str, Any]) -> None:
            await _emit(TraceEvent(agent_id=agent_id, run_group_id=run_group_id, step=step, timestamp=time.time(), detail=detail))

        # 1. Run Agent1
        async def _run_agent1() -> TurnMetrics:
            t0 = time.perf_counter()
            llm_before = self.llm.snapshot()
            jev_before = self.jev.snapshot()

            async def _step(step: str, detail: Dict[str, Any]) -> None:
                await _emit_trace("agent1", step, detail)

            await _emit(ChatEvent(agent_id="agent1", run_group_id=run_group_id, event="status", data={"status": "running"}))
            out = await self.agent1_graph.ainvoke(
                {
                    "session_id": session_id,
                    "run_group_id": run_group_id,
                    "user_message": user_message,
                    "event_callback": _step,
                }
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            trace = AgentTurnTrace(**out.get("trace", {"agent_id": "agent1"}))
            llm_delta = self.llm.delta(llm_before)
            jev_delta = self.jev.delta(jev_before)

            if trace.tools_called > 0:
                await _emit(
                    ChatEvent(
                        agent_id="agent1",
                        run_group_id=run_group_id,
                        event="tool_call",
                        data={"tool": trace.details.get("tool_name", "tool"), "latency_ms": trace.details.get("tool_latency_ms", 0)},
                    )
                )

            await _emit(ChatEvent(agent_id="agent1", run_group_id=run_group_id, event="final", data={"answer": trace.answer}))

            metrics = build_turn_metrics(
                trace=trace,
                llm_delta=llm_delta,
                jev_delta=jev_delta,
                latency_ms_total=elapsed_ms,
                session_id=session_id,
                user_message=user_message,
                run_group_id=run_group_id,
                turn_index=turn_index,
                model_name=self.llm.model_name,
                store_backends={"postgres": self.agent1_memory.facts_store.backend},
            )
            self.metrics_store.record_turn(metrics)
            await _emit(ChatEvent(agent_id="agent1", run_group_id=run_group_id, event="metrics", data=metrics.model_dump()))
            await _emit_trace("agent1", "turn_complete", metrics.model_dump())
            return metrics

        # 2. Run Agent2
        async def _run_agent2() -> TurnMetrics:
            t0 = time.perf_counter()
            llm_before = self.llm.snapshot()
            jev_before = self.jev.snapshot()

            async def _step(step: str, detail: Dict[str, Any]) -> None:
                await _emit_trace("agent2", step, detail)

            await _emit(ChatEvent(agent_id="agent2", run_group_id=run_group_id, event="status", data={"status": "running"}))
            out = await self.agent2_graph.graph.ainvoke(
                {
                    "session_id": session_id,
                    "run_group_id": run_group_id,
                    "user_message": user_message,
                    "event_callback": _step,
                }
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            trace = AgentTurnTrace(**out.get("trace", {"agent_id": "agent2"}))
            llm_delta = self.llm.delta(llm_before)
            jev_delta = self.jev.delta(jev_before)

            # STM append stays on the fast path (§4.2): the *next* turn's history
            # can't wait on the background graph-write below to finish.
            self.agent2_memory.append_turn(session_id, user_message, trace.answer)

            if trace.tools_called > 0:
                await _emit(
                    ChatEvent(
                        agent_id="agent2",
                        run_group_id=run_group_id,
                        event="tool_call",
                        data={"tool": trace.details.get("tool_name", "tool"), "latency_ms": trace.details.get("tool_latency_ms", 0)},
                    )
                )

            await _emit(ChatEvent(agent_id="agent2", run_group_id=run_group_id, event="final", data={"answer": trace.answer}))

            metrics = build_turn_metrics(
                trace=trace,
                llm_delta=llm_delta,
                jev_delta=jev_delta,
                latency_ms_total=elapsed_ms,
                session_id=session_id,
                user_message=user_message,
                run_group_id=run_group_id,
                turn_index=turn_index,
                model_name=self.llm.model_name,
                store_backends={"neo4j": self.agent2_memory.graph_store.backend},
            )
            self.metrics_store.record_turn(metrics)
            await _emit(ChatEvent(agent_id="agent2", run_group_id=run_group_id, event="metrics", data=metrics.model_dump()))
            await _emit_trace("agent2", "turn_complete", metrics.model_dump())

            # Fire-and-forget (§4.2): fact extraction + JEV staleness + graph write
            # never gates the answer above. `out` already carries everything the
            # background step needs (session_id, user_message, run_group_id).
            task = asyncio.create_task(self._run_agent2_background(out, session_id, run_group_id, _emit))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            return metrics

        # 3. Concurrently run both agents (§8 asyncio.gather)
        m1, m2 = await asyncio.gather(_run_agent1(), _run_agent2())

        # 4. Broadcast the graph as it stands right now — agent2's fact
        # extraction/staleness-check/write runs decoupled in the background
        # (§4.2), so this snapshot won't include this turn's writes yet; the
        # background task's own "graph_updated" event (below) tells GraphView
        # to refetch once those writes actually land.
        graph_data = await self.agent2_memory.graph_snapshot(session_id)
        await _emit(ChatEvent(agent_id="agent2", run_group_id=run_group_id, event="graph", data=graph_data))

        # 5. Broadcast session summary update
        summary = self.metrics_store.get_session_summary(session_id)
        await _emit(ChatEvent(agent_id="all", run_group_id=run_group_id, event="summary", data=summary))

        return {
            "run_group_id": run_group_id,
            "agent1": m1.model_dump(),
            "agent2": m2.model_dump(),
            "graph": graph_data,
            "summary": summary,
        }

    async def _run_agent2_background(
        self,
        state: Agent2State,
        session_id: str,
        run_group_id: str,
        emit: EventCallback,
    ) -> None:
        """The decoupled fact-extraction + JEV staleness + graph-write step
        (docs/IMPLEMENTATION_V2.md §4.2) — kicked off after agent2's answer has
        already been sent to the user, never gating it."""
        await emit(TraceEvent(agent_id="agent2", run_group_id=run_group_id, step="graph_write_start", timestamp=time.time(), detail={}))
        t0 = time.perf_counter()
        jev_before = self.jev.snapshot()
        try:
            result: BackgroundGraphWriteResult = await self.agent2_graph.run_background_graph_write(state)
        except Exception as exc:  # a failed background write must never crash the loop or vanish silently
            await emit(
                ChatEvent(
                    agent_id="agent2",
                    run_group_id=run_group_id,
                    event="graph_updated",
                    data={"nodes_added": 0, "edges_added": 0, "stale_marked": 0, "error": f"{type(exc).__name__}: {exc}"},
                )
            )
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        jev_calls = int(self.jev.delta(jev_before).get("total_calls", 0))
        self.metrics_store.record_graph_write(session_id, run_group_id, elapsed_ms, jev_calls)

        await emit(
            ChatEvent(
                agent_id="agent2",
                run_group_id=run_group_id,
                event="graph_updated",
                data={
                    "nodes_added": result.facts_added,
                    "edges_added": result.facts_added + result.facts_superseded,
                    "stale_marked": result.facts_superseded,
                    "latency_ms": round(elapsed_ms, 2),
                    "jev_calls": jev_calls,
                },
            )
        )
