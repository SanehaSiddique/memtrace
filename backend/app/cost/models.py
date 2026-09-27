"""Cost-domain models: one RunRecord per answered query, plus the aggregates
built from them for the executive dashboard."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

LEAK_LABELS = {
    "outdated_information": "Outdated information",
    "unverified_information": "Unverified information",
    "redundant_memory": "Redundant memory",
    "irrelevant_context": "Irrelevant information",
}

# Categories a memory decision can be attributed to on the Memory ROI table.
# The first four map directly to the product's four example rows; the fifth
# ("unverified_information") is a real, distinct category our pipeline
# produces (a PENDING_REVIEW memory retrieved then held back) and is shown
# alongside them rather than force-merged into an inexact bucket.
MEMORY_ROI_LABELS = {
    "outdated_information": "Removed stale context",
    "deduplicated_memory": "Deduplicated memory",
    "irrelevant_context": "Excluded irrelevant memory",
    "archived_obsolete": "Archived obsolete memory",
    "unverified_information": "Unverified information held back",
}


class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    agent_id: str
    conversation_id: str
    query: str
    model: str
    baseline_tokens: int
    optimized_tokens: int
    output_tokens: int
    baseline_cost: float
    optimized_cost: float
    savings: float
    leak_category: str = "none"
    langsmith_run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LeakBreakdownItem(BaseModel):
    category: str
    label: str
    amount: float
    percent: float


class CostSummary(BaseModel):
    total_runs: int
    lifetime_savings: float
    savings_this_month: float
    savings_last_month: float
    savings_change_vs_last_month: float
    projected_annual_savings: float
    memory_driven_savings: float
    total_cost_without_memtrace: float
    total_cost_with_memtrace: float
    avg_savings_per_run: float
    days_active: int
    data_source_note: str


class TimeseriesPoint(BaseModel):
    date: str
    daily_savings: float
    cumulative_savings: float


class MemoryCostImpact(BaseModel):
    """One memory decision's estimated dollar effect — the atomic unit the
    Memory ROI table and per-memory 'cost impact' displays are built from."""

    id: str = Field(default_factory=lambda: f"mci_{uuid.uuid4().hex[:10]}")
    agent_id: str
    memory_id: str
    operation: str  # a MEMORY_ROI_LABELS key
    run_id: Optional[str] = None
    cost_avoided: float
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryRoiRow(BaseModel):
    operation: str
    label: str
    event_count: int
    cost_avoided: float


class MemoryRoiSummary(BaseModel):
    rows: List[MemoryRoiRow]
    total_cost_avoided: float
    data_source_note: str


# ---------- Executive "Revenue & Model Mix" projection ----------

# Horizon in days, used to turn a measured per-run token volume into a period
# total. Weekly/monthly/annual are all requested by the CEO view, so they are
# named once here rather than spelled as magic numbers at each call site.
HORIZONS = {"weekly": 7, "monthly": 30, "annual": 365}


class UsageProfile(BaseModel):
    """Measured per-run token volumes, averaged across real recorded runs.

    This is what makes the projection *precise* rather than an average: the
    numbers come from `run_costs` rows the pipeline actually wrote (real
    retrieved-vs-selected context sizes), not a hand-typed "$0.08 per run".
    """

    baseline_input_tokens: int
    optimized_input_tokens: int
    output_tokens: int
    measured_runs: int
    is_measured: bool  # False => fell back to a documented default
    data_source_note: str


class ModelProjectionRow(BaseModel):
    """One model's cost at the requested fleet size, across all three horizons."""

    model: str
    label: str
    provider: str
    input_per_1k: float
    output_per_1k: float
    weekly_gross: float
    monthly_gross: float
    annual_gross: float
    weekly_with_memtrace: float
    monthly_with_memtrace: float
    annual_with_memtrace: float
    weekly_savings: float
    monthly_savings: float
    annual_savings: float
    savings_pct: float
    # Signed difference against the model the user currently runs. Negative
    # means this model is cheaper than the incumbent.
    monthly_delta_vs_current: float
    annual_delta_vs_current: float


class RevenueProjectionRequest(BaseModel):
    agent_id: Optional[str] = None
    agents: int = 100
    runs_per_agent_per_day: int = 500
    current_model: str = "gpt-4o-mini"
    compare_models: List[str] = Field(default_factory=list)
    # Revenue per agent per month, so the view can express AI spend as a
    # share of revenue and savings as margin points rather than only dollars.
    revenue_per_agent_month: float = 0.0


class RevenueProjection(BaseModel):
    agents: int
    runs_per_agent_per_day: int
    current_model: str
    runs_per_day: int
    weekly_runs: int
    monthly_runs: int
    annual_runs: int
    profile: UsageProfile
    rows: List[ModelProjectionRow]
    # CEO framing (only meaningful when revenue_per_agent_month > 0).
    monthly_revenue: float
    annual_revenue: float
    current_model_annual_spend_pct_of_revenue: float
    best_model: Optional[str] = None
    best_model_annual_savings: float = 0.0
    margin_points_recovered: float = 0.0
    data_source_note: str


def empty_summary(note: str) -> CostSummary:
    return CostSummary(
        total_runs=0,
        lifetime_savings=0.0,
        savings_this_month=0.0,
        savings_last_month=0.0,
        savings_change_vs_last_month=0.0,
        projected_annual_savings=0.0,
        memory_driven_savings=0.0,
        total_cost_without_memtrace=0.0,
        total_cost_with_memtrace=0.0,
        avg_savings_per_run=0.0,
        days_active=0,
        data_source_note=note,
    )
