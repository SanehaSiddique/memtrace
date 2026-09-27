"""MemoryService: the ingestion-side orchestrator.

EVENT -> extraction -> judgment -> consolidation -> graph update

This is deliberately plain async Python, not a LangGraph node itself — the
LangGraph ingest workflow (app/agent/workflow.py) wraps these same calls as
small, traceable nodes. Keeping the logic here means it's directly unit
testable without spinning up a graph.
"""

from typing import List

from app.judgment.interface import BaseMemoryJudge
from app.llm.interface import BaseLLMClient
from app.memory.consolidation import apply_operation
from app.memory.extraction import extract_candidates
from app.memory.models import MemoryEvent, MemoryOperationRecord
from app.memory.repository import BaseMemoryRepository
from app.memory.retrieval import find_linkable_active_memories

DEFAULT_SUBJECT = "Project Alpha"


class MemoryService:
    def __init__(
        self,
        repository: BaseMemoryRepository,
        llm_client: BaseLLMClient,
        judge: BaseMemoryJudge,
        default_subject: str = DEFAULT_SUBJECT,
    ) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self.judge = judge
        self.default_subject = default_subject

    async def ingest_event(self, event: MemoryEvent) -> List[MemoryOperationRecord]:
        await self.repository.save_event(event)

        candidates = await extract_candidates(event, self.default_subject, self.llm_client)

        records: List[MemoryOperationRecord] = []
        for candidate in candidates:
            existing_active = await find_linkable_active_memories(
                self.repository,
                self.llm_client,
                event.agent_id,
                candidate.subject,
                candidate.predicate,
                candidate.object,
                candidate.content,
            )
            judgment = await self.judge.judge(candidate, existing_active, event)
            record = await apply_operation(self.repository, self.llm_client, judgment, candidate, event)
            records.append(record)

        return records
