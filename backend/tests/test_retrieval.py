"""Hybrid retrieval + temporal filtering + context building, end to end on the
classic MongoDB -> PostgreSQL supersession scenario."""

import pytest

from app.context.builder import build_context
from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient
from app.memory.models import MemoryEvent
from app.memory.repository import SQLiteMemoryRepository
from app.memory.retrieval import apply_temporal_filter, hybrid_retrieve
from app.memory.service import MemoryService


@pytest.fixture
async def seeded_repo(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "retrieval.db"))
    await repo.initialize()
    llm = MockLLMClient()
    service = MemoryService(repo, llm, MockMemoryJudge(), default_subject="Project Alpha")
    await service.ingest_event(MemoryEvent(conversation_id="c", content="Project Alpha uses MongoDB."))
    await service.ingest_event(
        MemoryEvent(
            conversation_id="c",
            content="We migrated from MongoDB to PostgreSQL because relational querying became important.",
        )
    )
    return repo, llm


@pytest.mark.asyncio
async def test_current_fact_query_excludes_historical(seeded_repo):
    repo, llm = seeded_repo
    query = "What database are we currently using?"

    scored = await hybrid_retrieve(repo, llm, query, "agent-alpha")
    selected, excluded = apply_temporal_filter(query, scored)

    selected_objects = {sm.memory.object for sm in selected}
    assert "PostgreSQL" in selected_objects
    assert "MongoDB" not in selected_objects

    excluded_objects = {em.memory.object: em.reason for em in excluded}
    assert "MongoDB" in excluded_objects
    assert "Superseded" in excluded_objects["MongoDB"]
    assert "PostgreSQL" in excluded_objects["MongoDB"]

    context = build_context(query, selected, excluded)
    assert "PostgreSQL" in context.context_text
    assert context.token_estimate > 0


@pytest.mark.asyncio
async def test_historical_intent_query_keeps_both(seeded_repo):
    repo, llm = seeded_repo
    query = "What database did we use before PostgreSQL?"

    scored = await hybrid_retrieve(repo, llm, query, "agent-alpha")
    selected, _ = apply_temporal_filter(query, scored)

    selected_objects = {sm.memory.object for sm in selected}
    assert "MongoDB" in selected_objects
    assert "PostgreSQL" in selected_objects


@pytest.mark.asyncio
async def test_keyword_score_favors_real_overlap_not_short_memories(tmp_path):
    """Regression test: a short, barely-relevant memory used to outrank a longer,
    genuinely on-topic one whenever a query shared only generic terms (like the
    subject name) with both, because the old formula normalized by memory length."""
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "keyword_bug.db"))
    await repo.initialize()
    llm = MockLLMClient()
    service = MemoryService(repo, llm, MockMemoryJudge(), default_subject="Project Alpha")

    await service.ingest_event(MemoryEvent(conversation_id="c", content="Project Alpha uses AWS."))
    await service.ingest_event(
        MemoryEvent(
            conversation_id="c",
            content="We migrated from MongoDB to PostgreSQL because relational querying became important.",
        )
    )

    query = "Which database does Project Alpha use?"
    scored = await hybrid_retrieve(repo, llm, query, "agent-alpha")
    top = max(scored, key=lambda sm: sm.score)
    assert top.memory.object == "PostgreSQL"
