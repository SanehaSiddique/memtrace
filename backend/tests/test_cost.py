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

    # Memory ROI: the excluded MongoDB memory should show up attributed to
    # "Removed stale context", independent of the run-level aggregate above.
    roi = await cost_service.memory_roi_summary("agent-alpha")
    stale_row = next(r for r in roi.rows if r.operation == "outdated_information")
    assert stale_row.event_count == 1
    assert stale_row.cost_avoided > 0
    assert roi.total_cost_avoided == pytest.approx(stale_row.cost_avoided)

    mongo_memory_id = excluded[0].memory.id
    assert await cost_service.cost_impact_for_memory(mongo_memory_id) == pytest.approx(stale_row.cost_avoided)

    summary_with_roi = await cost_service.summary("agent-alpha")
    assert summary_with_roi.memory_driven_savings == pytest.approx(stale_row.cost_avoided)


@pytest.mark.asyncio
async def test_record_lifecycle_impact_for_merge_and_archive(tmp_path):
    cost_repo = SQLiteCostRepository(db_path=str(tmp_path / "cost2.db"))
    await cost_repo.initialize()
    cost_service = CostService(cost_repo)

    merge_impact = await cost_service.record_lifecycle_impact(
        agent_id="agent-alpha",
        operation="MERGE",
        memory_id="mem_1",
        content="Project Alpha uses MongoDB.",
        reason="Restates already-active memory.",
        model="gpt-4o-mini",
    )
    assert merge_impact is not None
    assert merge_impact.operation == "deduplicated_memory"
    assert merge_impact.cost_avoided > 0

    archive_impact = await cost_service.record_lifecycle_impact(
        agent_id="agent-alpha",
        operation="ARCHIVE",
        memory_id="mem_2",
        content="Some obsolete fact.",
        reason="No longer relevant.",
        model="gpt-4o-mini",
    )
    assert archive_impact.operation == "archived_obsolete"

    # ADD/UPDATE/REVIEW/NOOP aren't lifecycle cost events — nothing to record.
    assert await cost_service.record_lifecycle_impact(
        agent_id="agent-alpha", operation="ADD", memory_id="mem_3", content="x", reason="", model="gpt-4o-mini"
    ) is None

    roi = await cost_service.memory_roi_summary("agent-alpha")
    by_op = {r.operation: r for r in roi.rows}
    assert by_op["deduplicated_memory"].event_count == 1
    assert by_op["archived_obsolete"].event_count == 1
    assert by_op["irrelevant_context"].event_count == 0  # present with zero, not missing


@pytest.mark.asyncio
async def test_timeseries_granularity_buckets_by_month(tmp_path):
    from datetime import datetime, timezone

    from app.cost.models import RunRecord

    cost_repo = SQLiteCostRepository(db_path=str(tmp_path / "cost3.db"))
    await cost_repo.initialize()
    cost_service = CostService(cost_repo)

    for day in (1, 15, 28):
        await cost_repo.record(
            RunRecord(
                agent_id="agent-alpha",
                conversation_id="c",
                query="q",
                model="gpt-4o-mini",
                baseline_tokens=10,
                optimized_tokens=5,
                output_tokens=2,
                baseline_cost=0.01,
                optimized_cost=0.005,
                savings=0.005,
                created_at=datetime(2026, 1, day, tzinfo=timezone.utc),
            )
        )

    daily = await cost_service.timeseries("agent-alpha", granularity="day")
    assert len(daily) == 3

    monthly = await cost_service.timeseries("agent-alpha", granularity="month")
    assert len(monthly) == 1
    assert monthly[0].date == "2026-01-01"
    assert monthly[0].cumulative_savings == pytest.approx(0.015)


def test_project_scale_is_pure_arithmetic():
    projection = project_scale(agents=100, runs_per_agent_per_day=500, cost_per_run=0.08, avoidable_pct=22.0)
    assert projection.daily_savings == pytest.approx(100 * 500 * 0.08 * 0.22)
    assert projection.monthly_savings == pytest.approx(projection.daily_savings * 30)
    assert projection.annual_savings == pytest.approx(projection.daily_savings * 365)
