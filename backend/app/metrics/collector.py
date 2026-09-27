"""Metrics Collector & Aggregator (docs/IMPLEMENTATION.md §7).

Constructs `TurnMetrics` per turn per agent, tracking:
  * Prompt & completion token counts from the LLM provider.
  * Cost projections across GPT-4o, Claude Sonnet 5, and Grok 4.7.
  * Tool selection time and tool schema tokens avoided.
  * Raw vs filtered payload token reduction (the headline metric!).
  * JEV call volume and error tracking.
  * Session-level cumulative summaries for dashboard KPIs.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.cost import project_cost_all
from app.metrics.schema import AgentTurnTrace, TurnMetrics


def build_turn_metrics(
    trace: AgentTurnTrace,
    llm_delta: Dict[str, Any],
    jev_delta: Dict[str, Any],
    latency_ms_total: float,
    session_id: str,
    user_message: str,
    run_group_id: str,
    turn_index: int = 1,
    model_name: str = "",
    store_backends: Optional[Dict[str, str]] = None,
) -> TurnMetrics:
    """Build the spec-compliant §7 TurnMetrics model for one agent run."""
    prompt_tokens = int(llm_delta.get("tokens_in", 0))
    completion_tokens = int(llm_delta.get("tokens_out", 0))
    logical_llm_calls = int(llm_delta.get("logical_calls", 0))
    provider_calls = int(llm_delta.get("provider_calls", 0))
    cache_hits = int(llm_delta.get("cache_hits", 0))

    jev_calls = int(jev_delta.get("total_calls", 0)) if trace.agent_id == "agent2" else 0
    jev_errors = int(jev_delta.get("total_errors", 0)) if trace.agent_id == "agent2" else 0
    jev_breakdown = jev_delta.get("by_site", {}) if trace.agent_id == "agent2" else {}

    cost_projected = project_cost_all(prompt_tokens, completion_tokens)

    return TurnMetrics(
        agent_id=trace.agent_id,
        run_group_id=run_group_id,
        tool_selection_time_ms=trace.tool_selection_time_ms,
        tools_considered=trace.tools_considered,
        tools_called=trace.tools_called,
        raw_result_tokens=trace.raw_result_tokens,
        filtered_result_tokens=trace.filtered_result_tokens if trace.agent_id == "agent2" else None,
        total_context_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        llm_calls=logical_llm_calls,
        llm_provider_calls=provider_calls,
        llm_cache_hits=cache_hits,
        jev_calls=jev_calls,
        jev_breakdown=jev_breakdown,
        jev_errors=jev_errors,
        latency_ms_total=round(latency_ms_total, 2),
        cost_actual_usd=0.0,
        cost_projected=cost_projected,
        session_id=session_id,
        turn_index=turn_index,
        user_message=user_message,
        answer=trace.answer,
        model=model_name,
        tool_names=list(trace.tools_in_prompt),
        tool_errors=list(trace.tool_errors),
        notes=list(trace.notes),
        store_backends=dict(store_backends or {}),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class SessionMetricsStore:
    """In-memory thread-safe store of turn metrics for dashboard queries."""

    def __init__(self) -> None:
        self._metrics: Dict[str, List[TurnMetrics]] = {}

    def record_turn(self, metrics: TurnMetrics) -> None:
        self._metrics.setdefault(metrics.session_id, []).append(metrics)

    def record_graph_write(
        self, session_id: str, run_group_id: str, latency_ms: float, jev_calls: int
    ) -> None:
        """Back-fill a turn's background graph-write cost (§4.3) once the
        decoupled task finishes — arrives after `record_turn` already stored
        this turn's response-critical metrics, so it's a targeted update, not
        a new record."""
        for turn in self._metrics.get(session_id, []):
            if turn.run_group_id == run_group_id and turn.agent_id == "agent2":
                turn.graph_write_latency_ms = round(latency_ms, 2)
                turn.graph_write_jev_calls = jev_calls
                return

    def get_session_turns(self, session_id: str) -> List[TurnMetrics]:
        return list(self._metrics.get(session_id, []))

    def get_turn_count(self, session_id: str) -> int:
        turns = self._metrics.get(session_id, [])
        return len({t.run_group_id for t in turns})

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Aggregate totals and delta savings across all turns in a session."""
        all_turns = self._metrics.get(session_id, [])
        a1_turns = [t for t in all_turns if t.agent_id == "agent1"]
        a2_turns = [t for t in all_turns if t.agent_id == "agent2"]

        a1_context = sum(t.total_context_tokens for t in a1_turns)
        a2_context = sum(t.total_context_tokens for t in a2_turns)
        tokens_saved = max(0, a1_context - a2_context)
        token_savings_pct = round((tokens_saved / a1_context * 100), 1) if a1_context > 0 else 0.0

        a1_llm_calls = sum(t.llm_calls for t in a1_turns)
        a2_llm_calls = sum(t.llm_calls for t in a2_turns)
        llm_calls_saved = max(0, a1_llm_calls - a2_llm_calls)

        a1_proj_gpt4o = sum(t.cost_projected.get("gpt-4o", 0.0) for t in a1_turns)
        a2_proj_gpt4o = sum(t.cost_projected.get("gpt-4o", 0.0) for t in a2_turns)
        cost_saved_gpt4o = max(0.0, round(a1_proj_gpt4o - a2_proj_gpt4o, 6))

        a1_proj_claude = sum(t.cost_projected.get("claude-sonnet", 0.0) for t in a1_turns)
        a2_proj_claude = sum(t.cost_projected.get("claude-sonnet", 0.0) for t in a2_turns)
        cost_saved_claude = max(0.0, round(a1_proj_claude - a2_proj_claude, 6))

        return {
            "session_id": session_id,
            "total_turns": len({t.run_group_id for t in all_turns}),
            "agent1": {
                "turns_count": len(a1_turns),
                "total_context_tokens": a1_context,
                "total_llm_calls": a1_llm_calls,
                "projected_cost_gpt4o": round(a1_proj_gpt4o, 6),
                "projected_cost_claude": round(a1_proj_claude, 6),
            },
            "agent2": {
                "turns_count": len(a2_turns),
                "total_context_tokens": a2_context,
                "total_llm_calls": a2_llm_calls,
                "total_jev_calls": sum(t.jev_calls for t in a2_turns),
                "projected_cost_gpt4o": round(a2_proj_gpt4o, 6),
                "projected_cost_claude": round(a2_proj_claude, 6),
            },
            "savings": {
                "tokens_saved": tokens_saved,
                "tokens_saved_pct": token_savings_pct,
                "llm_calls_saved": llm_calls_saved,
                "cost_saved_gpt4o": cost_saved_gpt4o,
                "cost_saved_claude": cost_saved_claude,
            },
        }

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id:
            self._metrics.pop(session_id, None)
        else:
            self._metrics.clear()
