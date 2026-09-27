"""LangGraph workflow smoke tests: ingest graph then query graph on the
MongoDB -> PostgreSQL scenario, end to end."""

import pytest

from app.agent.nodes import make_generate_response_node
from app.agent.workflow import build_ingest_workflow, build_query_workflow
from app.context.builder import build_context
from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient
from app.memory.models import MemoryEvent, MemoryOperationType
from app.memory.repository import SQLiteMemoryRepository


class GeneralHelperLLM(MockLLMClient):
    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        assert "general-purpose personal assistant" in system
        assert "answer from general knowledge" in system
        assert "(no memory met" in user
        return "Paris is the capital of France."


@pytest.mark.asyncio
async def test_ingest_then_query_workflow(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "workflow.db"))
    await repo.initialize()
    llm = MockLLMClient()
    judge = MockMemoryJudge()

    ingest_graph = build_ingest_workflow(repo, llm, judge, default_subject="Project Alpha")

    result1 = await ingest_graph.ainvoke({"event": MemoryEvent(conversation_id="c", content="Project Alpha uses MongoDB.")})
    assert result1["operations"][0].operation == MemoryOperationType.ADD

    result2 = await ingest_graph.ainvoke({
        "event": MemoryEvent(
            conversation_id="c",
            content="We migrated from MongoDB to PostgreSQL because relational querying became important.",
        )
    })
    assert result2["operations"][0].operation == MemoryOperationType.UPDATE

    query_graph = build_query_workflow(repo, llm)
    result = await query_graph.ainvoke(
        {
            "query": "What database are we currently using?",
            "conversation_id": "c",
            "agent_id": "agent-alpha",
            "recent_messages": [],
        }
    )

    assert "PostgreSQL" in result["answer"]
    assert "MongoDB" not in result["answer"]
    selected_objects = {sm.memory.object for sm in result["selected_memories"]}
    assert selected_objects == {"PostgreSQL"}
    excluded_objects = {em.memory.object for em in result["excluded_memories"]}
    assert "MongoDB" in excluded_objects
    assert result["context"].token_estimate > 0
    assert result["trace_metadata"]["selected_memory_count"] == 1


@pytest.mark.asyncio
async def test_answer_node_handles_general_questions_without_memory():
    node = make_generate_response_node(GeneralHelperLLM())
    result = await node(
        {
            "context": build_context("What is the capital of France?", [], []),
            "selected_memories": [],
            "excluded_memories": [],
            "agent_id": "agent-alpha",
            "conversation_id": "conversation-1",
        }
    )

    assert result["answer"] == "Paris is the capital of France."
    assert result["trace_metadata"]["selected_memory_count"] == 0
