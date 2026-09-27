"""Applies a MemoryJudgment to the memory graph.

This is where the "never silently overwrite historical information" rule
lives: UPDATE always preserves the old memory (marked HISTORICAL) and links
it to the new one with SUPERSEDES/REPLACED_BY edges instead of mutating the
old row's content.
"""

from typing import List, Optional

from app.llm.interface import BaseLLMClient
from app.memory.models import (
    CandidateMemory,
    Memory,
    MemoryEvent,
    MemoryOperationRecord,
    MemoryOperationType,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
    RelationType,
)
from app.memory.repository import BaseMemoryRepository

_REVIEW_MEMORY_TYPES = {MemoryType.GOAL, MemoryType.EVENT_SUMMARY}


def _embedding_text(subject: str, predicate: str, obj: str, content: str) -> str:
    """Embed the structured triple alongside the raw sentence so semantic search isn't
    purely at the mercy of exact wording overlap."""
    return f"{subject} {predicate.replace('_', ' ')} {obj}. {content}"


def _initial_status(memory_type: MemoryType, confidence: float) -> MemoryStatus:
    if confidence < 0.5:
        return MemoryStatus.PENDING_REVIEW
    if memory_type in _REVIEW_MEMORY_TYPES and confidence < 0.7:
        return MemoryStatus.PENDING_REVIEW
    return MemoryStatus.ACTIVE


async def apply_operation(
    repository: BaseMemoryRepository,
    llm_client: BaseLLMClient,
    judgment,
    candidate: CandidateMemory,
    event: MemoryEvent,
) -> MemoryOperationRecord:
    # Use the event's own timestamp, not wall-clock time, so a memory's created_at/
    # valid_from/valid_until reflect when the underlying event happened — this matters
    # for backdated/simulated events (e.g. the demo story) as much as live ones.
    now = event.timestamp

    if judgment.operation == MemoryOperationType.NOOP:
        return MemoryOperationRecord(
            operation=judgment.operation,
            source_event_id=event.event_id,
            judgment=judgment,
            reason=judgment.reason,
        )

    if judgment.operation == MemoryOperationType.ARCHIVE and judgment.target_memory_id:
        archived = await repository.archive_memory(judgment.target_memory_id, reason=judgment.reason)
        return MemoryOperationRecord(
            operation=judgment.operation,
            memory_id=archived.id if archived else None,
            target_memory_id=judgment.target_memory_id,
            source_event_id=event.event_id,
            judgment=judgment,
            reason=judgment.reason,
        )

    if judgment.operation == MemoryOperationType.DELETE and judgment.target_memory_id:
        deleted = await repository.delete_memory(judgment.target_memory_id, reason=judgment.reason)
        return MemoryOperationRecord(
            operation=judgment.operation,
            memory_id=deleted.id if deleted else None,
            target_memory_id=judgment.target_memory_id,
            source_event_id=event.event_id,
            judgment=judgment,
            reason=judgment.reason,
        )

    if judgment.operation == MemoryOperationType.MERGE and judgment.target_memory_id:
        existing = await repository.get_memory(judgment.target_memory_id)
        if existing is None:
            return MemoryOperationRecord(
                operation=MemoryOperationType.NOOP,
                source_event_id=event.event_id,
                judgment=judgment,
                reason="MERGE target no longer exists.",
            )
        existing.confidence = max(existing.confidence, judgment.confidence)
        existing.provenance = {
            **existing.provenance,
            "reinforced_by_events": [*existing.provenance.get("reinforced_by_events", []), event.event_id],
        }
        await repository.update_memory(existing)
        await repository.add_relationship(
            MemoryRelationship(
                source_id=event.event_id,
                source_type="event",
                target_id=existing.id,
                target_type="memory",
                relation_type=RelationType.CAUSED_BY,
                metadata={"operation": "MERGE"},
            )
        )
        return MemoryOperationRecord(
            operation=judgment.operation,
            memory_id=existing.id,
            target_memory_id=existing.id,
            source_event_id=event.event_id,
            judgment=judgment,
            reason=judgment.reason,
        )

    # ADD, UPDATE, REVIEW all create a new memory row; UPDATE additionally retires the old one.
    embedding = await llm_client.embed(
        _embedding_text(judgment.subject, judgment.predicate, judgment.object, judgment.content)
    )
    new_memory = Memory(
        agent_id=event.agent_id,
        memory_type=judgment.memory_type,
        subject=judgment.subject,
        predicate=judgment.predicate,
        object=judgment.object,
        content=judgment.content,
        confidence=judgment.confidence,
        status=MemoryStatus.PENDING_REVIEW if judgment.operation == MemoryOperationType.REVIEW else _initial_status(judgment.memory_type, judgment.confidence),
        created_at=now,
        valid_from=now,
        source_event_id=event.event_id,
        provenance={
            "event_content": event.content,
            "extraction_reason": candidate.extraction_reason,
            "judgment_reason": judgment.reason,
        },
    )

    if judgment.operation == MemoryOperationType.UPDATE and judgment.target_memory_id:
        old_memory = await repository.get_memory(judgment.target_memory_id)
        if old_memory is not None:
            # Inherit the target's (subject, predicate) rather than whatever the
            # extractor worded this candidate's as — a live LLM extractor may phrase
            # the same real-world fact differently turn to turn (e.g. "We" vs "Project
            # Alpha"); linking is already confirmed semantically by the judge, so this
            # keeps the lineage keyed consistently without a rule-based normalizer.
            new_memory.subject = old_memory.subject
            new_memory.predicate = old_memory.predicate
            old_memory.status = MemoryStatus.HISTORICAL
            old_memory.valid_until = now
            old_memory.superseded_by_memory_id = new_memory.id
            await repository.update_memory(old_memory)
            new_memory.supersedes_memory_id = old_memory.id

    await repository.create_memory(new_memory, embedding=embedding)

    await repository.add_relationship(
        MemoryRelationship(
            source_id=event.event_id,
            source_type="event",
            target_id=new_memory.id,
            target_type="memory",
            relation_type=RelationType.CAUSED_BY,
            metadata={"operation": judgment.operation.value},
        )
    )

    if judgment.operation == MemoryOperationType.UPDATE and new_memory.supersedes_memory_id:
        await repository.add_relationship(
            MemoryRelationship(
                source_id=new_memory.supersedes_memory_id,
                source_type="memory",
                target_id=new_memory.id,
                target_type="memory",
                relation_type=RelationType.REPLACED_BY,
            )
        )
        await repository.add_relationship(
            MemoryRelationship(
                source_id=new_memory.id,
                source_type="memory",
                target_id=new_memory.supersedes_memory_id,
                target_type="memory",
                relation_type=RelationType.SUPERSEDES,
            )
        )

    return MemoryOperationRecord(
        operation=judgment.operation,
        memory_id=new_memory.id,
        target_memory_id=judgment.target_memory_id,
        source_event_id=event.event_id,
        judgment=judgment,
        reason=judgment.reason,
    )
