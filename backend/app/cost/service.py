"""Turns actual pipeline output (what MEMTRACE retrieved vs. what it actually
sent to the model) into dollar figures the rest of the app can aggregate.

Every number here is derived from a real recorded run and a configured price
table — never a hardcoded "we save X%" claim. `CostSummary.data_source_note`
always says so explicitly, per the product rule: simulated/projected numbers
must be labeled as such, not presented as verified production billing.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from app.context.builder import estimate_tokens
from app.cost.models import (
    HORIZONS,
    LEAK_LABELS,
    MEMORY_ROI_LABELS,
    CostSummary,
    LeakBreakdownItem,
    MemoryCostImpact,
    MemoryRoiRow,
    MemoryRoiSummary,
    ModelProjectionRow,
    RevenueProjection,
    RunRecord,
    TimeseriesPoint,
    UsageProfile,
    empty_summary,
)
from app.cost.pricing import get_pricing
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


def _bucket_key_fn(granularity: str):
    """Returns a function truncating a date to the requested bucket, as an ISO
    label: a day as-is, a week as its Monday, a month as its 1st, a year as
    its Jan 1st."""
    if granularity == "week":
        return lambda d: (d - timedelta(days=d.weekday())).isoformat()
    if granularity == "month":
        return lambda d: date(d.year, d.month, 1).isoformat()
    if granularity == "year":
        return lambda d: date(d.year, 1, 1).isoformat()
    return lambda d: d.isoformat()


class CostService:
    def __init__(self, repository) -> None:
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
        await self.repository.record(record)

        # Attribute cost to the individual memory decisions that made up this
        # run's savings — this is what the Memory ROI table and per-memory
        # "cost impact" displays are built from, independent of (and more
        # granular than) the single aggregate `savings` figure above.
        for item in excluded:
            memory_cost = (estimate_tokens(item.memory.content) / 1000) * pricing.input_per_1k
            await self.repository.record_impact(
                MemoryCostImpact(
                    agent_id=agent_id,
                    memory_id=item.memory.id,
                    operation=_categorize_excluded_reason(item.reason),
                    run_id=record.id,
                    cost_avoided=memory_cost,
                    reason=item.reason,
                )
            )

        return record

    async def record_lifecycle_impact(
        self,
        agent_id: str,
        operation: str,
        memory_id: str,
        content: str,
        reason: str,
        model: str,
    ) -> Optional[MemoryCostImpact]:
        """Attribute cost avoided to a MERGE/ARCHIVE/DELETE lifecycle decision made
        at ingestion time — e.g. a MERGE means a duplicate memory (and every future
        run that would have retrieved it) never needed to exist at all."""
        category = {"MERGE": "deduplicated_memory", "ARCHIVE": "archived_obsolete", "DELETE": "archived_obsolete"}.get(
            operation
        )
        if category is None:
            return None

        pricing = get_pricing(model)
        cost_avoided = (estimate_tokens(content) / 1000) * pricing.input_per_1k
        impact = MemoryCostImpact(
            agent_id=agent_id,
            memory_id=memory_id,
            operation=category,
            run_id=None,
            cost_avoided=cost_avoided,
            reason=reason,
        )
        return await self.repository.record_impact(impact)

    async def summary(self, agent_id: str) -> CostSummary:
        runs = await self.repository.list_all(agent_id)
        impacts = await self.repository.list_impacts(agent_id)
        memory_driven_savings = sum(i.cost_avoided for i in impacts)

        if not runs:
            summary = empty_summary("No runs recorded yet — seed demo data or ask a question to generate activity.")
            summary.memory_driven_savings = memory_driven_savings
            return summary

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
            memory_driven_savings=memory_driven_savings,
            total_cost_without_memtrace=total_without,
            total_cost_with_memtrace=total_with,
            avg_savings_per_run=lifetime_savings / len(runs),
            days_active=days_active,
            data_source_note=(
                f"Demo projection based on {len(runs)} recorded run(s) over {days_active} day(s) of activity "
                "and configured model pricing — not live provider billing data."
            ),
        )

    async def timeseries(self, agent_id: str, granularity: str = "day") -> List[TimeseriesPoint]:
        runs = await self.repository.list_all(agent_id)
        if not runs:
            return []

        bucket = _bucket_key_fn(granularity)
        by_bucket: dict = defaultdict(float)
        for run in runs:
            by_bucket[bucket(run.created_at.date())] += run.savings

        points: List[TimeseriesPoint] = []
        cumulative = 0.0
        for key in sorted(by_bucket.keys()):
            cumulative += by_bucket[key]
            points.append(TimeseriesPoint(date=key, daily_savings=by_bucket[key], cumulative_savings=cumulative))
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

    async def memory_roi_summary(self, agent_id: str) -> MemoryRoiSummary:
        """Connects memory lifecycle decisions directly to AI cost savings —
        "which memory decisions are saving us money?" """
        impacts = await self.repository.list_impacts(agent_id)

        totals: dict = defaultdict(float)
        counts: dict = defaultdict(int)
        for impact in impacts:
            totals[impact.operation] += impact.cost_avoided
            counts[impact.operation] += 1

        rows = [
            MemoryRoiRow(
                operation=category,
                label=label,
                event_count=counts.get(category, 0),
                cost_avoided=totals.get(category, 0.0),
            )
            for category, label in MEMORY_ROI_LABELS.items()
        ]
        rows.sort(key=lambda r: r.cost_avoided, reverse=True)

        return MemoryRoiSummary(
            rows=rows,
            total_cost_avoided=sum(totals.values()),
            data_source_note=(
                f"Calculated from {len(impacts)} recorded memory decision(s) and configured model pricing "
                "— demo data, not live provider billing."
            ),
        )

    async def cost_impact_for_memory(self, memory_id: str) -> float:
        impacts = await self.repository.list_impacts_for_memory(memory_id)
        return sum(i.cost_avoided for i in impacts)


# Fallback per-run token volumes, used only when no runs have been recorded
# yet. These are the same shape `record_run` measures (baseline context,
# optimized context, answer) so switching to measured data later is a drop-in
# change rather than a different calculation.
DEFAULT_BASELINE_TOKENS = 4000
DEFAULT_OPTIMIZED_TOKENS = 1200
DEFAULT_OUTPUT_TOKENS = 300


def _round2(value: float) -> float:
    return round(value + 0.0, 2)


async def usage_profile(repository, agent_id: str) -> UsageProfile:
    """Average the token volumes the pipeline actually recorded for this agent.

    This is what makes the projection precise rather than an average: the
    numbers come from real `run_costs` rows (context the retriever surfaced
    before the memory filter vs. context actually sent to the model), not a
    hand-typed "$0.08 per run".
    """
    runs = await repository.list_all(agent_id)
    if not runs:
        return UsageProfile(
            baseline_input_tokens=DEFAULT_BASELINE_TOKENS,
            optimized_input_tokens=DEFAULT_OPTIMIZED_TOKENS,
            output_tokens=DEFAULT_OUTPUT_TOKENS,
            measured_runs=0,
            is_measured=False,
            data_source_note=(
                "No recorded runs yet — using a documented default per-run token profile "
                "(4k baseline context, 1.2k selected context, 300 answer tokens). "
                "Run the benchmark to replace this with measured volumes."
            ),
        )

    n = len(runs)
    return UsageProfile(
        baseline_input_tokens=round(sum(r.baseline_tokens for r in runs) / n),
        optimized_input_tokens=round(sum(r.optimized_tokens for r in runs) / n),
        output_tokens=round(sum(r.output_tokens for r in runs) / n),
        measured_runs=n,
        is_measured=True,
        data_source_note=(
            f"Measured across {n} recorded run(s): the average context the retriever surfaced "
            "before the memory filter vs. the context actually sent to the model."
        ),
    )


def model_projection(model: str, profile: UsageProfile, runs_per_period: dict) -> ModelProjectionRow:
    """Price one model at a fleet size, for all three horizons at once."""
    pricing = get_pricing(model)
    # Output tokens are billed on every run regardless of memory filtering, so
    # they are identical in the gross and with-MEMTRACE columns; only the input
    # side shrinks.
    output_cost_per_run = (profile.output_tokens / 1000) * pricing.output_per_1k
    gross_input_per_run = (profile.baseline_input_tokens / 1000) * pricing.input_per_1k
    optimized_input_per_run = (profile.optimized_input_tokens / 1000) * pricing.input_per_1k

    row = ModelProjectionRow(
        model=model,
        label=pricing.label or model,
        provider=pricing.provider,
        input_per_1k=pricing.input_per_1k,
        output_per_1k=pricing.output_per_1k,
        weekly_gross=0.0,
        monthly_gross=0.0,
        annual_gross=0.0,
        weekly_with_memtrace=0.0,
        monthly_with_memtrace=0.0,
        annual_with_memtrace=0.0,
        weekly_savings=0.0,
        monthly_savings=0.0,
        annual_savings=0.0,
        savings_pct=0.0,
        monthly_delta_vs_current=0.0,
        annual_delta_vs_current=0.0,
    )

    for horizon, run_count in runs_per_period.items():
        gross = (gross_input_per_run + output_cost_per_run) * run_count
        with_memtrace = (optimized_input_per_run + output_cost_per_run) * run_count
        setattr(row, f"{horizon}_gross", _round2(gross))
        setattr(row, f"{horizon}_with_memtrace", _round2(with_memtrace))
        setattr(row, f"{horizon}_savings", _round2(gross - with_memtrace))

    row.savings_pct = _round2((row.monthly_savings / row.monthly_gross * 100) if row.monthly_gross > 0 else 0.0)
    return row


async def revenue_projection(
    repository,
    agent_id: str,
    agents: int,
    runs_per_agent_per_day: int,
    current_model: str,
    compare_models: Optional[List[str]] = None,
    revenue_per_agent_month: float = 0.0,
) -> RevenueProjection:
    """Fleet economics for the Executive view: agents x model x horizon.

    Nothing here is a hand-entered average. The token volume is measured from
    recorded runs, the rate is the model's published price, and the deltas are
    arithmetic differences between those — so raising the agent count or
    switching OpenAI -> Claude moves the number for a traceable reason.
    """
    agents = max(1, int(agents))
    runs_per_agent_per_day = max(1, int(runs_per_agent_per_day))

    profile = await usage_profile(repository, agent_id)
    runs_per_day = agents * runs_per_agent_per_day
    runs_per_period = {h: runs_per_day * d for h, d in HORIZONS.items()}

    # Always price the incumbent plus whatever the caller asked to compare.
    # Unknown names still get a row (flagged as unpriced) rather than silently
    # vanishing from the comparison.
    models: List[str] = [current_model]
    for m in compare_models or []:
        if m and m not in models:
            models.append(m)

    rows = [model_projection(m, profile, runs_per_period) for m in models]
    current_row = next(r for r in rows if r.model == current_model)
    for r in rows:
        r.monthly_delta_vs_current = _round2(r.monthly_with_memtrace - current_row.monthly_with_memtrace)
        r.annual_delta_vs_current = _round2(r.annual_with_memtrace - current_row.annual_with_memtrace)

    # "Best" = lowest annual spend *with* MEMTRACE, but only if it actually
    # beats the incumbent. Reporting a "winner" that costs more would be
    # dishonest, so in that case the incumbent is the answer.
    cheapest = min(rows, key=lambda r: r.annual_with_memtrace)
    best_model: Optional[str] = None
    best_savings = 0.0
    if cheapest.model != current_model and cheapest.annual_with_memtrace < current_row.annual_with_memtrace:
        best_model = cheapest.model
        best_savings = current_row.annual_with_memtrace - cheapest.annual_with_memtrace

    monthly_revenue = agents * revenue_per_agent_month
    annual_revenue = monthly_revenue * 12
    margin_points = round((best_savings / annual_revenue) * 100, 3) if (annual_revenue > 0 and best_model) else 0.0

    return RevenueProjection(
        agents=agents,
        runs_per_agent_per_day=runs_per_agent_per_day,
        current_model=current_model,
        runs_per_day=runs_per_day,
        weekly_runs=runs_per_period["weekly"],
        monthly_runs=runs_per_period["monthly"],
        annual_runs=runs_per_period["annual"],
        profile=profile,
        rows=rows,
        monthly_revenue=_round2(monthly_revenue),
        annual_revenue=_round2(annual_revenue),
        current_model_annual_spend_pct_of_revenue=(
            round((current_row.annual_with_memtrace / annual_revenue) * 100, 3) if annual_revenue > 0 else 0.0
        ),
        best_model=best_model,
        best_model_annual_savings=_round2(best_savings),
        margin_points_recovered=margin_points,
        data_source_note=profile.data_source_note + " Figures are a configured-price projection, not live provider billing.",
    )
