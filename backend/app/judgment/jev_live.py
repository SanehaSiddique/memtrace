"""JEV via the real jevtypesafeai.com REST API (a `jv_live_...` key), the
confirmed-live production endpoint — distinct from both the `typesafe-sdk`
package's default `api.typesafe.ai` host (`jev.py`) and the Vercel AI Gateway
substitute (`ai_gateway.py`). Response payloads are validated against Jev's
*actually observed* wire shape (verified with a real manual call, not vendor
docs — those turned out to describe a different shape entirely) instead of
raw dict indexing, so a missing or reshaped field surfaces as a clean
fallback rather than a `KeyError` crash. Every call is a LangSmith span (see
`app.tracing.langsmith`), same as every LLM call in this pipeline.
"""

from typing import Dict, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field, ValidationError

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
from app.tracing.langsmith import traced


class JevNoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class JevChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)


class JevScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    legend: Optional[Dict[str, str]] = None


JevAnswer = Union[JevNoulAnswer, JevChoiceAnswer, JevScoreAnswer]


class JevUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    credits_remaining_usd: Optional[float] = None


class JevApiResponse(BaseModel):
    model: str
    answers: Dict[str, JevAnswer]
    usage: Optional[JevUsage] = None


def _choice_question(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


class LiveJEVClient(BaseMemoryJudge):
    is_live = True

    def __init__(self, api_key: str, url: str, model: Optional[str] = None, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def _post(self, state, questions: dict, model: Optional[str]) -> dict:
        # Traced with only `state`/`questions`/`model` as inputs — `self._api_key`
        # is read via closure, never passed as an arg, so it can't end up
        # serialized into the LangSmith trace.
        @traced(name="jev.decide", run_type="chain")
        async def _call(state, questions, model):
            body: dict = {"state": state, "questions": questions}
            if model:
                body["model"] = model
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, headers=self._headers(), json=body)
                response.raise_for_status()
                return response.json()

        return await _call(state, questions, model)

    async def _decide(self, state, questions: dict) -> Optional[JevApiResponse]:
        try:
            raw = await self._post(state, questions, self._model)
            return JevApiResponse.model_validate(raw)
        except (httpx.HTTPError, ValidationError, ValueError):
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
            target_criteria = {mid: f"{m.subject} {m.predicate} {m.object}: {m.content}" for mid, m in by_id.items()}
            target_criteria["none"] = "None of the listed memories are the same real-world fact as the new one."
            questions["target"] = _choice_question(
                "If UPDATE/MERGE/ARCHIVE/DELETE, which existing memory (by id) does this target? "
                "Pick 'none' for ADD/REVIEW/NOOP.",
                target_criteria,
            )

        result = await self._decide(state, questions)
        if result is None:
            return self._fallback(candidate, "jev_fallback: live JEV call failed or was unparseable")

        operation_answer = result.answers.get("operation")
        memory_type_answer = result.answers.get("memory_type")
        if not isinstance(operation_answer, JevChoiceAnswer) or not isinstance(memory_type_answer, JevChoiceAnswer):
            return self._fallback(candidate, "jev_fallback: expected choice answers from live JEV")

        try:
            operation = MemoryOperationType(operation_answer.choice)
            memory_type = MemoryType(memory_type_answer.choice)
        except ValueError:
            return self._fallback(candidate, "jev_fallback: unrecognized choice from live JEV")

        target_memory_id = None
        relation_to_target = None
        if operation in (
            MemoryOperationType.UPDATE,
            MemoryOperationType.MERGE,
            MemoryOperationType.ARCHIVE,
            MemoryOperationType.DELETE,
        ):
            target_answer = result.answers.get("target")
            target_choice = target_answer.choice if isinstance(target_answer, JevChoiceAnswer) else None
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
            confidence=operation_answer.confidence,
            target_memory_id=target_memory_id,
            relation_to_target=relation_to_target,
            reason=f"jev(live):{result.model}",
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
