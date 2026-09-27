"""Per-turn metrics for the comparison demo (docs/IMPLEMENTATION.md §7).

`TurnMetrics` is the contract between backend and frontend — the dashboard
renders directly off it, per turn, per agent, side by side. The fields and
their meanings are copied from the spec; nothing extra is required to be
displayed, and nothing here is estimated unless it says so (`*_tokens` for raw
tool payloads are ~4 chars/token estimates because no provider ever counted
them; LLM token counts are the provider's real reported usage).
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentTurnTrace(BaseModel):
    """Raw per-agent facts gathered while a turn runs.

    This is what the *agent* knows about its own turn; the collector adds the
    wrapped LLM/JEV accounting deltas to produce the final `TurnMetrics`.
    """

    agent_id: str
    agent_version: str = "v1"
    answer: str = ""
    used_tool: bool = False
    tools_considered: int = 0
    tools_called: int = 0
    tool_selection_time_ms: float = 0.0
    tools_in_prompt: List[str] = Field(default_factory=list)
    raw_result_tokens: Optional[int] = None
    filtered_result_tokens: Optional[int] = None
    tool_errors: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class TurnMetrics(BaseModel):
    """Exactly the spec's §7 payload."""

    agent_id: str  # "agent1" | "agent2"
    run_group_id: str  # correlates both agents' runs for this turn
    tool_selection_time_ms: float
    tools_considered: int  # how many tool schemas were in the prompt
    tools_called: int
    raw_result_tokens: Optional[int] = None  # None for agent1 if no tool called
    filtered_result_tokens: Optional[int] = None  # None for agent1 always (no filtering step)
    total_context_tokens: int = 0  # full prompt token count for the turn
    llm_calls: int = 0  # count of actual LLM calls this turn
    jev_calls: int = 0  # 0 for agent1 always
    latency_ms_total: float = 0.0
    cost_actual_usd: float = 0.0  # cost on the free model actually used (near-zero)
    cost_projected: Dict[str, float] = Field(default_factory=dict)  # {"gpt-4o": x, "claude-sonnet": y, "grok": z}

    # --- honest additions the dashboard needs, all measured -------------------
    session_id: str = ""
    turn_index: int = 0
    user_message: str = ""
    answer: str = ""
    model: str = ""
    tool_names: List[str] = Field(default_factory=list)
    llm_provider_calls: int = 0
    llm_cache_hits: int = 0
    completion_tokens: int = 0
    jev_breakdown: Dict[str, int] = Field(default_factory=dict)
    jev_errors: int = 0
    tool_errors: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    store_backends: Dict[str, str] = Field(default_factory=dict)
    created_at: str = ""

    # --- background graph-write (docs/IMPLEMENTATION_V2.md §4.3, agent2 only) --
    # Populated after the fact once the decoupled background task finishes —
    # None until then. Kept separate from `latency_ms_total`/`jev_calls` above
    # (which now reflect only the response-critical path) so the dashboard can
    # show where the time/JEV-call cost of the background write went, honestly,
    # rather than making agent2 look artificially fast/cheap by omission.
    graph_write_latency_ms: Optional[float] = None
    graph_write_jev_calls: Optional[int] = None

    # --- live "thinking process" data, carried into the stored record too --
    # (docs/IMPLEMENTATION_V2.md §6, agent2 only) — the TraceEvent stream (§5.2)
    # shows these live during the turn; these fields let the dashboard show the
    # same numbers after the fact, from `/api/metrics/{session_id}`.
    jev_routing_probabilities: Optional[Dict[str, float]] = None
    jev_filtering_chunk_scores: Optional[List[float]] = None


class TraceEvent(BaseModel):
    """One step in the live "thinking process" stream (docs/IMPLEMENTATION_V2.md
    §5.2) — a granular, in-flight look at what a turn is doing (JEV's routing
    probabilities, per-chunk filtering scores, streamed answer tokens, ...),
    distinct from the turn-level `ChatEvent`s (status/final/metrics) V1 already
    sends. Sent over the same `/ws/chat` connection; the frontend tells them
    apart by shape (`step`+`timestamp`+`detail` vs. `event`+`data`)."""

    agent_id: str  # "agent1" | "agent2"
    run_group_id: str
    step: str  # see the step vocabulary in docs/IMPLEMENTATION_V2.md §5.2
    timestamp: float
    detail: Dict[str, Any] = Field(default_factory=dict)


class TurnRequest(BaseModel):
    """What a client sends on `WS /ws/chat` (docs §8)."""

    message: str
    session_id: str = Field(default="default_session")


class ChatEvent(BaseModel):
    """One websocket event: `{agent_id, event, data}` (docs §8)."""

    agent_id: str
    event: str  # "token" | "tool_call" | "final" | "metrics" | "error" | "status"
    data: Dict[str, Any] = Field(default_factory=dict)
    run_group_id: str = ""


def new_run_group_id() -> str:
    return f"turn_{uuid.uuid4().hex[:12]}"
