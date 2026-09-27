"""The MEMTRACE demo story: an evolving fictional project, "Project Alpha",
authored so the timeline gives the memory graph real temporal structure —
old decisions that get superseded, a rejected alternative, a decision still
under consideration, and one deliberately vague statement that should land in
PENDING_REVIEW rather than ACTIVE.

Run directly (`python -m app.demo.seed`) to (re)populate MEMTRACE_DB_PATH.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

from app.config import settings
from app.judgment.factory import get_memory_judge
from app.llm.factory import get_llm_client
from app.memory.models import MemoryEvent, MemoryOperationRecord
from app.memory.repository import BaseMemoryRepository, SQLiteMemoryRepository
from app.memory.service import DEFAULT_SUBJECT, MemoryService

_START = datetime(2026, 1, 1, tzinfo=timezone.utc)

# (day offset, content) — the story, in order.
STORY: List[tuple[int, str]] = [
    (1, "Project Alpha uses MongoDB."),
    (5, "The team is considering PostgreSQL because relational querying would simplify reporting."),
    (8, "We migrated from MongoDB to PostgreSQL because relational querying became important."),
    (9, "The team rejected MySQL because it lacked the JSON support PostgreSQL offers."),
    (10, "We moved from AWS to Railway because it simplified our infrastructure."),
    (14, "We changed from session-based auth to JWT because it improved scalability."),
    (18, "The team selected Railway over Heroku because of simpler pricing."),
    (20, "We updated the PostgreSQL configuration to increase the connection pool size."),
    (22, "Project Alpha depends on Redis for caching."),
    (25, "We are considering GraphQL for the public API because clients want flexible queries."),
]


async def seed_demo_data(
    service: MemoryService,
    conversation_id: str = "demo_conversation",
    agent_id: str = "agent-alpha",
) -> List[MemoryOperationRecord]:
    all_records: List[MemoryOperationRecord] = []
    for day_offset, content in STORY:
        event = MemoryEvent(
            conversation_id=conversation_id,
            agent_id=agent_id,
            speaker="user",
            content=content,
            timestamp=_START + timedelta(days=day_offset),
            metadata={"day": day_offset},
        )
        records = await service.ingest_event(event)
        all_records.extend(records)
    return all_records


async def _main() -> None:
    repository: BaseMemoryRepository = SQLiteMemoryRepository(db_path=settings.memtrace_db_path)
    await repository.initialize()
    llm_client = get_llm_client(settings)
    judge = get_memory_judge(settings, llm_client)
    service = MemoryService(repository, llm_client, judge, default_subject=DEFAULT_SUBJECT)

    records = await seed_demo_data(service, agent_id=settings.memtrace_default_agent_id)
    for record in records:
        print(f"{record.operation.value:8s} memory={record.memory_id} target={record.target_memory_id} reason={record.reason}")


if __name__ == "__main__":
    asyncio.run(_main())
