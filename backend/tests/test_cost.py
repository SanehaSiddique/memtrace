"""The cost service turns real retrieval output into dollar figures. Verify
the math is honest: savings should be zero when nothing was excluded, and
positive when MEMTRACE actually trimmed stale/irrelevant memories out."""

import pytest

from app.cost.repository import SQLiteCostRepository
from app.cost.service import CostService, project_scale
from app.demo.seed import seed_demo_data
from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient
from app.memory.repository import SQLiteMemoryRepository
from app.memory.retrieval import apply_temporal_filter, score_and_rank, traverse_graph, retrieve_semantic
from app.memory.service import DEFAULT_SUBJECT, MemoryService


@pytest.mark.asyncio
async def test_record_run_reflects_real_exclusions(tmp_path):
    repo = SQLiteMemoryRepository(db_path=str(tmp_path / "mem.db"))
    await repo.initialize()
    llm = MockLLMClient()
    service = MemoryService(repo, llm, MockMemoryJudge(), default_subject=DEFAULT_SUBJECT)
    await seed_demo_data(service, agent_id="agent-alpha")

    cost_repo = SQLiteCostRepository(db_path=str(tmp_path / "cost.db"))
    await cost_repo.initialize()
    cost_service = CostService(cost_repo)

    query = "What database are we currently using?"
    candidates = await retrieve_semantic(repo, llm, query, "agent-alpha")
    candidates = await traverse_graph(repo, candidates)
    ranked = score_and_rank(candidates, top_k=8)
    selected, excluded = apply_temporal_filter(query, ranked)
    assert excluded  # the flagship case: MongoDB should be excluded

    run = await cost_service.record_run(
        agent_id="agent-alpha",
        conversation_id="c",
        query=query,
        model="gpt-4o-mini",
        retrieved=ranked,
        selected=selected,
        excluded=excluded,
        answer="Project Alpha uses database PostgreSQL.",
    )

    assert run.savings > 0
    assert run.optimized_tokens <= run.baseline_tokens
    assert run.leak_category == "outdated_information"

    summary = await cost_service.summary("agent-alpha")
    assert summary.total_runs == 1
    assert summary.lifetime_savings == pytest.approx(run.savings)
    assert "not live provider billing data" in summary.data_source_note

    leaks = await cost_service.leak_breakdown("agent-alpha")
    assert leaks[0].category == "outdated_information"
    assert leaks[0].percent == pytest.approx(100.0)

    timeseries = await cost_service.timeseries("agent-alpha")
    assert len(timeseries) == 1
    assert timeseries[0].cumulative_savings == pytest.approx(run.savings)


def test_project_scale_is_pure_arithmetic():
    projection = project_scale(agents=100, runs_per_agent_per_day=500, cost_per_run=0.08, avoidable_pct=22.0)
    assert projection.daily_savings == pytest.approx(100 * 500 * 0.08 * 0.22)
    assert projection.monthly_savings == pytest.approx(projection.daily_savings * 30)
    assert projection.annual_savings == pytest.approx(projection.daily_savings * 365)
