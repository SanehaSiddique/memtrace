"""JEV substitute over a plain chat LLM client (OpenRouter/OpenAI).

Fallback path used only when neither a real TypeSafe key nor the AI Gateway
JEV endpoint is configured but a live chat client is — same bounded
operation/memory_type criteria as `JEVMemoryJudge`/`AIGatewayJEVClient`,
forced into strict JSON and validated against the enums, since a plain chat
completion has no native bounded-choice primitive.
"""

import json
from typing import List

from app.judgment.interface import BaseMemoryJudge
from app.judgment.jev import _MEMORY_TYPE_CRITERIA, _OPERATION_CRITERIA
from app.llm.interface import BaseLLMClient
from app.memory.models import (
    CandidateMemory,
    Memory,
    MemoryEvent,
    MemoryJudgment,
    MemoryOperationType,
    MemoryType,
    RelationType,
)

_SYSTEM_PROMPT = (
    "You are a bounded classifier, not a chat assistant. Given a new candidate fact and any "
    "existing active memories that might cover the same thing (retrieval is deliberately "
    "recall-oriented — a few plausible candidates, not a guaranteed-correct one — so decide "
    "which, if any, is really the same real-world fact under different wording), respond with "
    'ONLY a JSON object: {"operation": one of ' + json.dumps(list(_OPERATION_CRITERIA)) + ', '
    '"memory_type": one of ' + json.dumps(list(_MEMORY_TYPE_CRITERIA)) + ', '
    '"target_id": the id of the existing memory this targets (required for UPDATE/MERGE/ARCHIVE/DELETE, '
    'else null), "confidence": float 0-1}. '
    "Operation meanings: " + json.dumps(_OPERATION_CRITERIA) + ". "
    "Memory type meanings: " + json.dumps(_MEMORY_TYPE_CRITERIA) + "."
)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.removeprefix("json").strip() if text.lower().startswith("json") else text
    return text


class LLMBoundedJudge(BaseMemoryJudge):
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm_client = llm_client

    async def judge(
        self,
        candidate: CandidateMemory,
        existing_active: List[Memory],
        event: MemoryEvent,
    ) -> MemoryJudgment:
        user_message = json.dumps(
            {
                "event_content": event.content,
                "new_fact": {
                    "subject": candidate.subject,
                    "predicate": candidate.predicate,
                    "object": candidate.object,
                    "content": candidate.content,
                },
                "existing_active_memories": [
                    {"id": m.id, "object": m.object, "content": m.content, "confidence": m.confidence}
                    for m in existing_active
                ],
            }
        )
        try:
            raw = await self._llm_client.chat(_SYSTEM_PROMPT, user_message)
            payload = json.loads(_strip_code_fence(raw))
            operation = MemoryOperationType(payload["operation"])
            memory_type = MemoryType(payload["memory_type"])
            confidence = float(payload.get("confidence", candidate.confidence))
        except Exception:
            return MemoryJudgment(
                operation=MemoryOperationType.ADD,
                memory_type=candidate.memory_type,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                content=candidate.content,
                confidence=candidate.confidence,
                reason="jev_fallback: LLM-bounded judge call failed or was unparseable",
            )

        target_memory_id = None
        relation_to_target = None
        if operation in (
            MemoryOperationType.UPDATE,
            MemoryOperationType.MERGE,
            MemoryOperationType.ARCHIVE,
            MemoryOperationType.DELETE,
        ):
            by_id = {m.id for m in existing_active}
            target_choice = payload.get("target_id")
            if target_choice in by_id:
                target_memory_id = target_choice
            if target_memory_id is None:
                operation = MemoryOperationType.ADD
            elif operation == MemoryOperationType.UPDATE:
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
            reason="jev(llm-bounded)",
        )
