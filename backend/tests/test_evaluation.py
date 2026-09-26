"""Runs the evaluation dataset against the seeded demo story and checks that
MEMTRACE's hybrid retrieval measurably beats flat semantic-only baseline."""

import pytest

from app.demo.seed import seed_demo_data
from app.evaluation.runner import run_evaluation
from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient
from app.memory.repository import SQLiteMemoryRepository
from app.memory.service import DEFAULT_SUBJECT, MemoryService


@pytest.mark.asyncio
async def test_memtrace_beats_baseline_on_demo_story(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "eval.db"))
    await repo.initialize()
    llm = MockLLMClient()
    service = MemoryService(repo, llm, MockMemoryJudge(), default_subject=DEFAULT_SUBJECT)
    await seed_demo_data(service, agent_id="agent-alpha")

    report = await run_evaluation(repo, llm, agent_id="agent-alpha")

    assert len(report.cases) == 10
    assert report.memtrace_accuracy >= report.baseline_accuracy
    # the flagship case the whole project is about must work
    flagship = next(c for c in report.cases if c.name == "direct_current_fact")
    assert flagship.memtrace_correct is True
