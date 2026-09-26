"""Turns actual pipeline output (what MEMTRACE retrieved vs. what it actually
sent to the model) into dollar figures the rest of the app can aggregate.

Every number here is derived from a real recorded run and a configured price
table — never a hardcoded "we save X%" claim. `CostSummary.data_source_note`
always says so explicitly, per the product rule: simulated/projected numbers
must be labeled as such, not presented as verified production billing.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from app.context.builder import estimate_tokens
from app.cost.models import (
    LEAK_LABELS,
    CostSummary,
    LeakBreakdownItem,
    RunRecord,
    ScaleProjection,
    TimeseriesPoint,
    empty_summary,
)
from app.cost.pricing import get_pricing
from app.cost.repository import SQLiteCostRepository
from app.memory.models import ExcludedMemory, ScoredMemory


def _categorize_excluded_reason(reason: str) -> str:
    lowered = reason.lower()
    if "pending review" in lowered or "not yet confirmed" in lowered:
        return "unverified_information"
    if "superseded" in lowered or "historical" in lowered or "archived" in lowered or "no longer currently valid" in lowered:
        return "outdated_information"
    if "deleted" in lowered:
        return "redundant_memory"
    return "irrelevant_context"


def _dominant_leak_category(excluded: List[ExcludedMemory]) -> str:
    if not excluded:
        return "none"
    counts: dict = {}
    for item in excluded:
        category = _categorize_excluded_reason(item.reason)
        counts[category] = counts.get(category, 0) + 1
    return max(counts, key=counts.get)


class CostService:
    def __init__(self, repository: SQLiteCostRepository) -> None:
        self.repository = repository

    async def initialize(self) -> None:
        await self.repository.initialize()

    async def record_run(
        self,
        agent_id: str,
        conversation_id: str,
        query: str,
        model: str,
        retrieved: List[ScoredMemory],
        selected: List[ScoredMemory],
        excluded: List[ExcludedMemory],
        answer: str,
        langsmith_run_id: Optional[str] = None,
    ) -> RunRecord:
        # "baseline" = every memory the hybrid retriever judged relevant BEFORE the
        # temporal/status filter ran — i.e. what a flat retrieval step would have
        # handed the model. "optimized" = what MEMTRACE actually kept.
        baseline_text = "\n".join(sm.memory.content for sm in retrieved) or query
        optimized_text = "\n".join(sm.memory.content for sm in selected) or ""

        baseline_tokens = max(estimate_tokens(baseline_text), estimate_tokens(optimized_text))
        optimized_tokens = estimate_tokens(optimized_text)
        output_tokens = estimate_tokens(answer)

        pricing = get_pricing(model)
        baseline_cost = (baseline_tokens / 1000) * pricing.input_per_1k + (output_tokens / 1000) * pricing.output_per_1k
        optimized_cost = (optimized_tokens / 1000) * pricing.input_per_1k + (output_tokens / 1000) * pricing.output_per_1k
        savings = max(0.0, baseline_cost - optimized_cost)

        record = RunRecord(
            agent_id=agent_id,
            conversation_id=conversation_id,
            query=query,
            model=model,
            baseline_tokens=baseline_tokens,
            optimized_tokens=optimized_tokens,
            output_tokens=output_tokens,
            baseline_cost=baseline_cost,
            optimized_cost=optimized_cost,
            savings=savings,
            leak_category=_dominant_leak_category(excluded),
            langsmith_run_id=langsmith_run_id,
        )
        return await self.repository.record(record)

    async def summary(self, agent_id: str) -> CostSummary:
        runs = await self.repository.list_all(agent_id)
        if not runs:
            return empty_summary("No runs recorded yet — seed demo data or ask a question to generate activity.")

        now = datetime.now(timezone.utc)
        lifetime_savings = sum(r.savings for r in runs)
        total_without = sum(r.baseline_cost for r in runs)
        total_with = sum(r.optimized_cost for r in runs)

        this_month = [r for r in runs if (r.created_at.year, r.created_at.month) == (now.year, now.month)]
        prev_month_index = now.month - 1 or 12
        prev_month_year = now.year if now.month > 1 else now.year - 1
        last_month = [r for r in runs if (r.created_at.year, r.created_at.month) == (prev_month_year, prev_month_index)]

        savings_this_month = sum(r.savings for r in this_month)
        savings_last_month = sum(r.savings for r in last_month)

        earliest = min(r.created_at for r in runs)
        days_active = max(1, (now - earliest).days + 1)
        daily_avg = lifetime_savings / days_active
        projected_annual = daily_avg * 365

        return CostSummary(
            total_runs=len(runs),
            lifetime_savings=lifetime_savings,
            savings_this_month=savings_this_month,
            savings_last_month=savings_last_month,
            savings_change_vs_last_month=savings_this_month - savings_last_month,
            projected_annual_savings=projected_annual,
            total_cost_without_memtrace=total_without,
            total_cost_with_memtrace=total_with,
            avg_savings_per_run=lifetime_savings / len(runs),
            days_active=days_active,
            data_source_note=(
                f"Demo projection based on {len(runs)} recorded run(s) over {days_active} day(s) of activity "
                "and configured model pricing — not live provider billing data."
            ),
        )

    async def timeseries(self, agent_id: str) -> List[TimeseriesPoint]:
        runs = await self.repository.list_all(agent_id)
        if not runs:
            return []

        by_day: dict = defaultdict(float)
        for run in runs:
            by_day[run.created_at.date().isoformat()] += run.savings

        points: List[TimeseriesPoint] = []
        cumulative = 0.0
        for date in sorted(by_day.keys()):
            cumulative += by_day[date]
            points.append(TimeseriesPoint(date=date, daily_savings=by_day[date], cumulative_savings=cumulative))
        return points

    async def leak_breakdown(self, agent_id: str) -> List[LeakBreakdownItem]:
        runs = await self.repository.list_all(agent_id)
        totals: dict = defaultdict(float)
        for run in runs:
            if run.leak_category == "none":
                continue
            totals[run.leak_category] += run.savings

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return []

        items = [
            LeakBreakdownItem(
                category=category,
                label=LEAK_LABELS.get(category, category),
                amount=amount,
                percent=(amount / grand_total) * 100,
            )
            for category, amount in totals.items()
        ]
        return sorted(items, key=lambda i: i.amount, reverse=True)

    async def recent_runs(self, agent_id: str, limit: int = 20) -> List[RunRecord]:
        return await self.repository.list_recent(agent_id, limit)


def project_scale(agents: int, runs_per_agent_per_day: int, cost_per_run: float, avoidable_pct: float) -> ScaleProjection:
    daily = agents * runs_per_agent_per_day * cost_per_run * (avoidable_pct / 100)
    return ScaleProjection(
        agents=agents,
        runs_per_agent_per_day=runs_per_agent_per_day,
        cost_per_run=cost_per_run,
        avoidable_pct=avoidable_pct,
        daily_savings=daily,
        monthly_savings=daily * 30,
        annual_savings=daily * 365,
    )
