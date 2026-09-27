"""Deterministic, rule-based MemoryJudge used whenever TYPESAFE_API_KEY is unset.

Keeps the full ADD/UPDATE/MERGE/REVIEW lifecycle real and testable without a
live JEV call. Swappable 1:1 for `JEVMemoryJudge` since both implement
`BaseMemoryJudge.judge()` with the same signature and return type.
"""

from typing import List

from app.judgment.interface import BaseMemoryJudge
from app.memory.models import (
    CandidateMemory,
    Memory,
    MemoryEvent,
    MemoryJudgment,
    MemoryOperationType,
    RelationType,
)

LOW_CONFIDENCE_THRESHOLD = 0.5


class MockMemoryJudge(BaseMemoryJudge):
    async def judge(
        self,
        candidate: CandidateMemory,
        existing_active: List[Memory],
        event: MemoryEvent,
    ) -> MemoryJudgment:
        if candidate.confidence < LOW_CONFIDENCE_THRESHOLD:
            return MemoryJudgment(
                operation=MemoryOperationType.REVIEW,
                memory_type=candidate.memory_type,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                content=candidate.content,
                confidence=candidate.confidence,
                reason="Low extraction confidence; routed to human review.",
            )

        # Retrieval is deliberately recall-oriented (a few plausible candidates, not
        # a guaranteed-correct one) and leaves disambiguation to the judge — but this
        # mock judge is deliberately simple (it always trusts the first candidate
        # rather than reasoning about it), so it only acts on an exact predicate
        # match, the one signal precise enough to trust blindly. A real judge (JEV)
        # is what handles the fuzzier candidates.
        same_predicate = [m for m in existing_active if m.predicate.strip().lower() == candidate.predicate.strip().lower()]
        if not same_predicate:
            return MemoryJudgment(
                operation=MemoryOperationType.ADD,
                memory_type=candidate.memory_type,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                content=candidate.content,
                confidence=candidate.confidence,
                reason="No existing active memory for this subject/predicate.",
            )

        existing = same_predicate[0]
        if existing.object.strip().lower() == candidate.object.strip().lower():
            return MemoryJudgment(
                operation=MemoryOperationType.MERGE,
                memory_type=candidate.memory_type,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                content=candidate.content,
                confidence=max(existing.confidence, candidate.confidence),
                target_memory_id=existing.id,
                reason=f"Restates already-active memory {existing.id}; reinforcing instead of duplicating.",
            )

        return MemoryJudgment(
            operation=MemoryOperationType.UPDATE,
            memory_type=candidate.memory_type,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            content=candidate.content,
            confidence=candidate.confidence,
            target_memory_id=existing.id,
            relation_to_target=RelationType.REPLACED_BY,
            reason=f"Supersedes memory {existing.id} ('{existing.object}' -> '{candidate.object}').",
        )
