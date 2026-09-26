"""End-to-end test of the ingestion pipeline: event -> extraction -> judgment -> consolidation."""

import pytest

from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient
from app.memory.models import MemoryEvent, MemoryOperationType, MemoryStatus
from app.memory.repository import SQLiteMemoryRepository
from app.memory.service import MemoryService


@pytest.mark.asyncio
async def test_migration_supersedes_old_fact(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "svc.db"))
    await repo.initialize()
    service = MemoryService(repo, MockLLMClient(), MockMemoryJudge(), default_subject="Project Alpha")

    e1 = MemoryEvent(conversation_id="conv_1", content="Project Alpha uses MongoDB.")
    records1 = await service.ingest_event(e1)
    assert len(records1) == 1
    assert records1[0].operation == MemoryOperationType.ADD

    active = await repo.get_active_memories("agent-alpha")
    assert len(active) == 1
    mongo_memory = active[0]
    assert mongo_memory.object == "MongoDB"
    assert mongo_memory.status == MemoryStatus.ACTIVE

    e2 = MemoryEvent(
        conversation_id="conv_1",
        content="We migrated from MongoDB to PostgreSQL because relational querying became important.",
    )
    records2 = await service.ingest_event(e2)
    assert len(records2) == 1
    assert records2[0].operation == MemoryOperationType.UPDATE

    active = await repo.get_active_memories("agent-alpha")
    assert len(active) == 1
    assert active[0].object == "PostgreSQL"
    assert active[0].supersedes_memory_id == mongo_memory.id

    historical = await repo.get_historical_memories("agent-alpha")
    assert len(historical) == 1
    assert historical[0].id == mongo_memory.id
    assert historical[0].superseded_by_memory_id == active[0].id

    history_chain = await repo.get_memory_history(active[0].id)
    assert [m.object for m in history_chain] == ["PostgreSQL", "MongoDB"]

    rels = await repo.get_relationships_for_node(mongo_memory.id)
    rel_types = {r.relation_type.value for r in rels}
    assert "REPLACED_BY" in rel_types


@pytest.mark.asyncio
async def test_repeating_same_fact_merges_instead_of_duplicating(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "svc2.db"))
    await repo.initialize()
    service = MemoryService(repo, MockLLMClient(), MockMemoryJudge(), default_subject="Project Alpha")

    await service.ingest_event(MemoryEvent(conversation_id="c", content="Project Alpha uses MongoDB."))
    records = await service.ingest_event(MemoryEvent(conversation_id="c", content="Project Alpha uses MongoDB."))

    assert records[0].operation == MemoryOperationType.MERGE
    active = await repo.get_active_memories("agent-alpha")
    assert len(active) == 1  # no duplicate row created


@pytest.mark.asyncio
async def test_low_confidence_extraction_routes_to_review(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "svc3.db"))
    await repo.initialize()
    service = MemoryService(repo, MockLLMClient(), MockMemoryJudge(), default_subject="Project Alpha")

    records = await service.ingest_event(
        MemoryEvent(conversation_id="c", content="Someone mentioned something vague in standup today.")
    )
    assert records[0].operation == MemoryOperationType.REVIEW
    pending = await repo.list_memories("agent-alpha", status=MemoryStatus.PENDING_REVIEW)
    assert len(pending) == 1
