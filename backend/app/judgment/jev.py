"""TypeSafe JEV ("System One") backed MemoryJudge.

Uses two bounded `Choice` questions — never free-form generation — to decide
the lifecycle operation and memory type. See the typesafe_sdk docs for the
System One primitives (Noul/Choice/Score); we only need Choice here since
each output is a pick from a fixed, small label set.
"""

from typing import List, Optional

from typesafe_sdk import AsyncTypeSafeClient, Choice

from app.judgment.interface import BaseMemoryJudge
from app.memory.models import (
    CandidateMemory,
    Memory,
    MemoryEvent,
    MemoryJudgment,
    MemoryOperationType,
    MemoryType,
    RelationType,
)

_OPERATION_CRITERIA = {
    "ADD": "No existing active memory covers this subject and predicate; create a new one.",
    "UPDATE": "An existing active memory covers this subject and predicate but with a different value; supersede it.",
    "MERGE": "An existing active memory already states this same value; reinforce it instead of duplicating.",
    "ARCHIVE": "The existing active memory is no longer relevant and should be archived with no replacement.",
    "DELETE": "The existing active memory was incorrect and should be removed outright.",
    "REVIEW": "The new fact is ambiguous or low-confidence and needs human review before becoming active.",
    "NOOP": "Nothing meaningful changed; take no action.",
}
_MEMORY_TYPE_CRITERIA = {
    "FACT": "An objective, stated fact.",
    "DECISION": "A choice the team made, with or without a stated alternative.",
    "PREFERENCE": "A stated preference that doesn't commit to a final decision.",
    "GOAL": "Something being considered or aimed for, not yet decided.",
    "CONSTRAINT": "A limitation or requirement that bounds future decisions.",
    "EVENT_SUMMARY": "A generic summary of an event that doesn't fit the other types.",
}


class JEVMemoryJudge(BaseMemoryJudge):
    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        self._client = AsyncTypeSafeClient(api_key=api_key, model=model)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def judge(
        self,
        candidate: CandidateMemory,
        existing_active: List[Memory],
        event: MemoryEvent,
    ) -> MemoryJudgment:
        state = {
            "event_content": event.content,
            "new_fact": {
                "subject": candidate.subject,
                "predicate": candidate.predicate,
                "object": candidate.object,
                "content": candidate.content,
            },
            "existing_active_memories_for_same_subject_predicate": [
                {"id": m.id, "object": m.object, "content": m.content, "confidence": m.confidence}
                for m in existing_active
            ],
        }
        result = await self._client.system_one(
            state=state,
            questions={
                "operation": Choice(
                    instructions="Which memory lifecycle operation should be applied to the new fact?",
                    criteria=_OPERATION_CRITERIA,
                ),
                "memory_type": Choice(
                    instructions="What kind of memory is the new fact?",
                    criteria=_MEMORY_TYPE_CRITERIA,
                ),
            },
        )
        operation = MemoryOperationType(result.choices["operation"].choice)
        memory_type = MemoryType(result.choices["memory_type"].choice)
        confidence = result.choices["operation"].confidence

        target_memory_id = None
        relation_to_target = None
        if operation in (MemoryOperationType.UPDATE, MemoryOperationType.MERGE, MemoryOperationType.ARCHIVE, MemoryOperationType.DELETE):
            if existing_active:
                target_memory_id = existing_active[0].id
            if operation == MemoryOperationType.UPDATE:
                relation_to_target = RelationType.REPLACED_BY

        return MemoryJudgment(
            operation=operation,
            memory_type=memory_type,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            content=candidate.content,
            confidence=confidence,
            target_memory_id=target_memory_id,
            relation_to_target=relation_to_target,
            reason=f"jev:{result.model}",
        )
