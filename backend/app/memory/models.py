"""Memory domain models and schemas for MEMTRACE."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    HISTORICAL = "HISTORICAL"
    ARCHIVED = "ARCHIVED"
    PENDING_REVIEW = "PENDING_REVIEW"
    DELETED = "DELETED"


class MemoryType(str, Enum):
    FACT = "FACT"
    DECISION = "DECISION"
    PREFERENCE = "PREFERENCE"
    GOAL = "GOAL"
    CONSTRAINT = "CONSTRAINT"
    EVENT_SUMMARY = "EVENT_SUMMARY"


class RelationType(str, Enum):
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    OWNS = "OWNS"
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    REPLACED_BY = "REPLACED_BY"
    SUPERSEDES = "SUPERSEDES"
    CAUSED_BY = "CAUSED_BY"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    PREFERS = "PREFERS"
    DECIDED = "DECIDED"
    CONTRADICTS = "CONTRADICTS"


class MemoryOperationType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    ARCHIVE = "ARCHIVE"
    DELETE = "DELETE"
    REVIEW = "REVIEW"
    NOOP = "NOOP"


class MemoryEvent(BaseModel):
    """Raw source event ingested by the memory layer."""
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    conversation_id: str = "default_conversation"
    agent_id: str = "agent-alpha"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    speaker: str = "user"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRelationship(BaseModel):
    """Graph edge between entities/memories."""
    id: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:10]}")
    source_id: str
    source_type: str = "memory"  # 'memory' or 'entity'
    target_id: str
    target_type: str = "memory"  # 'memory' or 'entity'
    relation_type: RelationType
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """Structured memory record."""
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}")
    agent_id: str = "agent-alpha"
    memory_type: MemoryType = MemoryType.FACT
    subject: str
    predicate: str
    object: str
    content: str
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    supersedes_memory_id: Optional[str] = None
    superseded_by_memory_id: Optional[str] = None
    source_event_id: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateMemory(BaseModel):
    """A candidate fact pulled out of a raw event, before judgment decides its lifecycle operation."""
    memory_type: MemoryType = MemoryType.FACT
    subject: str
    predicate: str
    object: str
    content: str
    confidence: float = 0.8
    extraction_reason: str = ""


class MemoryJudgment(BaseModel):
    """Bounded judgment result produced by MemoryJudge (e.g. JEV)."""
    operation: MemoryOperationType
    memory_type: MemoryType = MemoryType.FACT
    subject: str
    predicate: str
    object: str
    content: str
    confidence: float = 1.0
    target_memory_id: Optional[str] = None  # ID of memory to supersede, update, or archive
    relation_to_target: Optional[RelationType] = None
    reason: str = ""


class ScoredMemory(BaseModel):
    """Memory accompanied by explainable retrieval metrics."""
    memory: Memory
    score: float
    retrieval_reason: List[str]
    graph_distance: Optional[int] = None
    semantic_score: float = 0.0
    graph_score: float = 0.0
    temporal_score: float = 0.0
    status_score: float = 0.0


class ExcludedMemory(BaseModel):
    """A memory that was considered but left out of the final context."""
    memory: Memory
    reason: str


class MemoryOperationRecord(BaseModel):
    """Audit record of a single lifecycle operation applied to the memory graph."""
    id: str = Field(default_factory=lambda: f"op_{uuid.uuid4().hex[:10]}")
    operation: MemoryOperationType
    memory_id: Optional[str] = None
    target_memory_id: Optional[str] = None
    source_event_id: Optional[str] = None
    judgment: Optional[MemoryJudgment] = None
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextResult(BaseModel):
    """Output of the ContextBuilder: what the LLM actually saw, and why."""
    context_text: str
    selected: List[ScoredMemory] = Field(default_factory=list)
    excluded: List[ExcludedMemory] = Field(default_factory=list)
    token_estimate: int = 0


class GraphPathStep(BaseModel):
    """One hop in a graph traversal path, used for debug/explain output."""
    relationship: MemoryRelationship
    node_id: str
    node_label: str = ""


class DebugQueryResult(BaseModel):
    """Full explainability payload for `POST /debug/query`."""
    query: str
    conversation_id: str
    agent_id: str
    answer: str
    retrieved: List[ScoredMemory] = Field(default_factory=list)
    excluded: List[ExcludedMemory] = Field(default_factory=list)
    graph_paths: List[List[GraphPathStep]] = Field(default_factory=list)
    context: ContextResult
    langsmith_run_id: Optional[str] = None
    trace_metadata: Dict[str, Any] = Field(default_factory=dict)
