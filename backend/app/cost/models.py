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
    total_cost_without_memtrace: float
    total_cost_with_memtrace: float
    avg_savings_per_run: float
    days_active: int
    data_source_note: str


class TimeseriesPoint(BaseModel):
    date: str
    daily_savings: float
    cumulative_savings: float


class ScaleProjection(BaseModel):
    agents: int
    runs_per_agent_per_day: int
    cost_per_run: float
    avoidable_pct: float
    daily_savings: float
    monthly_savings: float
    annual_savings: float


class ScaleProjectionRequest(BaseModel):
    agents: int = 100
    runs_per_agent_per_day: int = 500
    cost_per_run: float = 0.08
    avoidable_pct: float = 22.0


def empty_summary(note: str) -> CostSummary:
    return CostSummary(
        total_runs=0,
        lifetime_savings=0.0,
        savings_this_month=0.0,
        savings_last_month=0.0,
        savings_change_vs_last_month=0.0,
        projected_annual_savings=0.0,
        total_cost_without_memtrace=0.0,
        total_cost_with_memtrace=0.0,
        avg_savings_per_run=0.0,
        days_active=0,
        data_source_note=note,
    )
