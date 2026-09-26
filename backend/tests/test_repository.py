"""Tests for BaseMemoryRepository and SQLiteMemoryRepository."""

import os
import pytest
from app.memory.models import (
    Memory,
    MemoryEvent,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
    RelationType,
)
from app.memory.repository import SQLiteMemoryRepository


@pytest.mark.asyncio
async def test_repository_events_and_memories(tmp_path):
    db_file = str(tmp_path / "test_memtrace.db")
    repo = SQLiteMemoryRepository(db_path=db_file)
    await repo.initialize()

    # Test event saving and fetching
    event = MemoryEvent(
        event_id="evt_001",
        conversation_id="conv_1",
        speaker="user",
        content="We migrated from MongoDB to PostgreSQL.",
    )
    saved_event = await repo.save_event(event)
    assert saved_event.event_id == "evt_001"

    fetched_event = await repo.get_event("evt_001")
    assert fetched_event is not None
    assert fetched_event.content == "We migrated from MongoDB to PostgreSQL."

    # Test memory creation
    mem1 = Memory(
        id="mem_001",
        agent_id="agent-alpha",
        memory_type=MemoryType.FACT,
        subject="Project Alpha",
        predicate="uses_database",
        object="MongoDB",
        content="Project Alpha uses MongoDB database.",
        status=MemoryStatus.HISTORICAL,
        source_event_id=event.event_id,
    )
    await repo.create_memory(mem1)

    mem2 = Memory(
        id="mem_002",
        agent_id="agent-alpha",
        memory_type=MemoryType.FACT,
        subject="Project Alpha",
        predicate="uses_database",
        object="PostgreSQL",
        content="Project Alpha uses PostgreSQL database.",
        status=MemoryStatus.ACTIVE,
        supersedes_memory_id="mem_001",
        source_event_id=event.event_id,
    )
    await repo.create_memory(mem2)

    # Fetch memory
    loaded_mem2 = await repo.get_memory("mem_002")
    assert loaded_mem2 is not None
    assert loaded_mem2.object == "PostgreSQL"
    assert loaded_mem2.status == MemoryStatus.ACTIVE
    assert loaded_mem2.supersedes_memory_id == "mem_001"

    # List memories
    active_mems = await repo.list_memories("agent-alpha", status=MemoryStatus.ACTIVE)
    assert len(active_mems) == 1
    assert active_mems[0].id == "mem_002"

    all_mems = await repo.list_memories("agent-alpha")
    assert len(all_mems) == 2

    # Graph relationship
    rel = MemoryRelationship(
        source_id="mem_001",
        source_type="memory",
        target_id="mem_002",
        target_type="memory",
        relation_type=RelationType.REPLACED_BY,
    )
    await repo.add_relationship(rel)

    rels = await repo.get_relationships_for_node("mem_001")
    assert len(rels) == 1
    assert rels[0].relation_type == RelationType.REPLACED_BY
    assert rels[0].target_id == "mem_002"

    # History traversal
    history = await repo.get_memory_history("mem_002")
    assert len(history) == 2
    assert history[0].id == "mem_002"
    assert history[1].id == "mem_001"
