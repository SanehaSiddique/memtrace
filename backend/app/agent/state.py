"""LangGraph execution state.

Deliberately holds only per-run scratch data — the query text, intermediate
retrieval results, the final answer. The persistent memory graph itself lives
in `BaseMemoryRepository`, never in this state.
"""

from typing import Any, Dict, List

from typing_extensions import TypedDict

from app.memory.models import (
    CandidateMemory,
    ContextResult,
    ExcludedMemory,
    MemoryEvent,
    MemoryOperationRecord,
    ScoredMemory,
)


class QueryState(TypedDict, total=False):
    query: str
    conversation_id: str
    agent_id: str
    recent_messages: List[str]
    wants_history: bool
    candidate_pool: Dict[str, ScoredMemory]
    retrieved_memories: List[ScoredMemory]
    selected_memories: List[ScoredMemory]
    excluded_memories: List[ExcludedMemory]
    context: ContextResult
    answer: str
    trace_metadata: Dict[str, Any]
    errors: List[str]


class IngestState(TypedDict, total=False):
    event: MemoryEvent
    candidates: List[CandidateMemory]
    operations: List[MemoryOperationRecord]
    trace_metadata: Dict[str, Any]
    errors: List[str]
