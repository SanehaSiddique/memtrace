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
from app.agent2.graph_cleaning import GraphCleaningResult, clean_recent_graph
from app.agent2.memory import Agent2Memory
from app.core.graph8_client import Graph8MCPClient
from app.core.jev_client import JEVClient
from app.core.llm_client import LLMClient
from app.metrics.collector import SessionMetricsStore, build_turn_metrics
from app.metrics.history import ChatHistoryRepository
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
        history: Optional[ChatHistoryRepository] = None,
        graph_clean_enabled: bool = True,
        graph_clean_window: int = 5,
        graph_clean_conflict_threshold: float = 0.6,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.jev = jev
        self.agent1_memory = agent1_memory
        self.agent2_memory = agent2_memory
        self.metrics_store = metrics_store or SessionMetricsStore()
        # Durable transcript. Optional so the orchestrator stays usable in tests
        # and scripts that don't care about persistence.
        self.history = history

        # Post-response graph cleaning (§ graph_cleaning.py)
        self.graph_clean_enabled = graph_clean_enabled
        self.graph_clean_window = graph_clean_window
        self.graph_clean_conflict_threshold = graph_clean_conflict_threshold

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

        async def _persist(
            agent_id: str,
            role: str,
            content: str,
            tool_calls: Optional[list] = None,
            metrics: Optional[Dict[str, Any]] = None,
        ) -> None:
            """Append one row to the durable transcript.

            Best-effort by design: a history write failing must never cost the
            user their answer, which has already been streamed by this point.
            """
            if self.history is None or not content:
                return
            try:
                await self.history.record(
                    session_id=session_id,
                    agent_id=agent_id,
                    run_group_id=run_group_id,
                    turn_index=turn_index,
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    metrics=metrics,
                )
            except Exception:
                pass

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
            # Persist the answer *with* its metrics so rehydrating the page
            # restores both the bubble and the per-turn figures from one row.
            await _persist(
                "agent1",
                "assistant",
                trace.answer,
                tool_calls=[{"tool": trace.details.get("tool_name", "tool"), "latency_ms": trace.details.get("tool_latency_ms", 0)}]
                if trace.tools_called > 0
                else [],
                metrics=metrics.model_dump(mode="json"),
            )
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

            # Persist the answer *with* its metrics so rehydrating the page
            # restores both the bubble and the per-turn figures from one row.
            await _persist(
                "agent2",
                "assistant",
                trace.answer,
                tool_calls=[{"tool": trace.details.get("tool_name", "tool"), "latency_ms": trace.details.get("tool_latency_ms", 0)}]
                if trace.tools_called > 0
                else [],
                metrics=metrics.model_dump(mode="json"),
            )

            return metrics

        # The prompt is stored once per agent so each panel's transcript is
        # self-contained and can be re-fetched without interleaving.
        await asyncio.gather(
            _persist("agent1", "user", user_message),
            _persist("agent2", "user", user_message),
        )

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

        # Post-response graph cleaning: the staleness pass above only compares a
        # *new* fact against same-entity actives, so contradictions that already
        # sit in the graph can survive it. Re-check the newest few nodes and
        # retire the older of any conflicting pair. Still fully decoupled from
        # the response path — the user already has their answer.
        clean_result = None
        if self.graph_clean_enabled:
            clean_result = await self._clean_graph(session_id, run_group_id, emit)

        await emit(
            ChatEvent(
                agent_id="agent2",
                run_group_id=run_group_id,
                event="graph_updated",
                data={
                    "nodes_added": result.facts_added,
                    "edges_added": result.facts_added + result.facts_superseded + result.relations_added,
                    "stale_marked": result.facts_superseded,
                    "relations_added": result.relations_added,
                    "latency_ms": round(elapsed_ms, 2),
                    "jev_calls": jev_calls,
                    "cleaned": (clean_result.nodes_marked_stale if clean_result else 0),
                    "cleaned_classes": (clean_result.class_counts if clean_result else {}),
                },
            )
        )

    async def _clean_graph(
        self,
        session_id: str,
        run_group_id: str,
        emit: EventCallback,
    ) -> Optional[GraphCleaningResult]:
        """Run the cleaning pass, degrading to "no cleaning" on any failure."""
        try:
            result = await clean_recent_graph(
                graph_store=self.agent2_memory.graph_store,
                jev=self.jev,
                session_id=session_id,
                window=self.graph_clean_window,
                conflict_threshold=self.graph_clean_conflict_threshold,
                run_group_id=run_group_id,
            )
        except Exception as exc:  # never let cleaning break the turn
            await emit(
                TraceEvent(
                    agent_id="agent2",
                    run_group_id=run_group_id,
                    step="graph_cleaning_failed",
                    timestamp=time.time(),
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            return None

        if result.changed:
            # Re-broadcast the graph so the UI picks up the newly-classified nodes.
            graph_data = await self.agent2_memory.graph_snapshot(session_id)
            await emit(ChatEvent(agent_id="agent2", run_group_id=run_group_id, event="graph", data=graph_data))
        return result
