"""Unit tests for the deterministic offline answer synthesizer."""

from app.agent.answering import synthesize_offline_answer
from app.memory.models import Memory, MemoryStatus, ScoredMemory


def _scored(memory: Memory) -> ScoredMemory:
    return ScoredMemory(memory=memory, score=1.0, retrieval_reason=["test"])


def test_no_selected_memory_gives_honest_fallback():
    assert synthesize_offline_answer([]) == "I don't have a currently valid memory to answer that."


def test_mentions_a_rejected_alternative_alongside_the_active_fact():
    active = Memory(
        subject="Project Alpha",
        predicate="uses_database",
        object="PostgreSQL",
        content="We use PostgreSQL.",
        status=MemoryStatus.ACTIVE,
    )
    rejected = Memory(
        subject="Project Alpha",
        predicate="rejected",
        object="MySQL",
        content="MySQL was rejected.",
        status=MemoryStatus.ACTIVE,
    )

    answer = synthesize_offline_answer([_scored(active), _scored(rejected)])

    assert "PostgreSQL" in answer
    assert "MySQL was considered and rejected." in answer
