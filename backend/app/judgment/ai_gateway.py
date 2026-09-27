"""JEV via the real Vercel AI Gateway `/v1/evaluate` endpoint.

Confirmed live (see JEV_URL): this endpoint speaks the exact same "System One"
wire contract TypeSafe's own SDK uses (`state` + `questions`, answers keyed by
question name with `.choice`/`.confidence`) — just gateway-hosted under
`model: "jev"` (canonical slug `typesafe-ai/jev`) with a Bearer AI_GATEWAY_API_KEY
instead of TYPESAFE_API_KEY. So this reuses the exact same bounded
operation/memory_type criteria `JEVMemoryJudge` already authored, just over a
plain httpx POST instead of the typesafe_sdk client.
"""

from typing import List, Optional

import httpx

from app.judgment.interface import BaseMemoryJudge
from app.judgment.jev import _MEMORY_TYPE_CRITERIA, _OPERATION_CRITERIA
from app.memory.models import (
    CandidateMemory,
    Memory,
    MemoryEvent,
    MemoryJudgment,
    MemoryOperationType,
    MemoryType,
    RelationType,
)

JEV_MODEL = "jev"


def _choice_question(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


class AIGatewayJEVClient(BaseMemoryJudge):
    is_live = True

    def __init__(self, api_key: str, url: str, model: str = JEV_MODEL, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def evaluate(self, state, questions: dict) -> Optional[dict]:
        """Raw System-One-shaped call, reused by the memory judge below and by
        other bounded yes/no/choice decisions elsewhere (e.g. the query
        pipeline's "does this need external lookup?" check) so every bounded
        decision in the app goes through the same real JEV endpoint. Returns
        None (never raises) on any HTTP/parse failure, so callers can fall
        back to a safe default instead of taking down the whole request."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    headers=self._headers(),
                    json={"model": self._model, "state": state, "questions": questions},
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            return None

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
        questions = {
            "operation": _choice_question(
                "Which memory lifecycle operation should be applied to the new fact?", _OPERATION_CRITERIA
            ),
            "memory_type": _choice_question("What kind of memory is the new fact?", _MEMORY_TYPE_CRITERIA),
        }
        by_id = {m.id: m for m in existing_active}
        if by_id:
            # Retrieval above is deliberately recall-oriented (a few plausible
            # candidates, not a guaranteed-correct single one) — JEV, not string
            # matching, decides which one (if any) is actually the same real-world
            # fact under a different wording.
            target_criteria = {mid: f"{m.subject} {m.predicate} {m.object}: {m.content}" for mid, m in by_id.items()}
            target_criteria["none"] = "None of the listed memories are the same real-world fact as the new one."
            questions["target"] = _choice_question(
                "If UPDATE/MERGE/ARCHIVE/DELETE, which existing memory (by id) does this target? "
                "Pick 'none' for ADD/REVIEW/NOOP.",
                target_criteria,
            )

        result = await self.evaluate(state, questions)
        if result is None:
            return self._fallback(candidate, "jev_fallback: AI Gateway call failed")

        try:
            answers = result["answers"]
            operation = MemoryOperationType(answers["operation"]["choice"])
            memory_type = MemoryType(answers["memory_type"]["choice"])
            confidence = float(answers["operation"].get("confidence", candidate.confidence))
        except (KeyError, ValueError, TypeError):
            return self._fallback(candidate, "jev_fallback: unparseable AI Gateway response")

        target_memory_id = None
        relation_to_target = None
        if operation in (
            MemoryOperationType.UPDATE,
            MemoryOperationType.MERGE,
            MemoryOperationType.ARCHIVE,
            MemoryOperationType.DELETE,
        ):
            target_choice = (answers.get("target") or {}).get("choice")
            if target_choice and target_choice in by_id:
                target_memory_id = target_choice
            if target_memory_id is None:
                operation = MemoryOperationType.ADD  # judge wanted to link but nothing valid to link to
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
            reason=f"jev(ai-gateway):{result.get('model', self._model)}",
        )

    def _fallback(self, candidate: CandidateMemory, reason: str) -> MemoryJudgment:
        return MemoryJudgment(
            operation=MemoryOperationType.ADD,
            memory_type=candidate.memory_type,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            content=candidate.content,
            confidence=candidate.confidence,
            reason=reason,
        )
