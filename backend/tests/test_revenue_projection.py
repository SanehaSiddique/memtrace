"""Revenue & model-mix projection (cost/service.py).

The point of these tests is the claim the Executive view now makes: the
figures are *measured*, not averaged. Each test pins one way that claim could
be quietly broken.
"""

import pytest

from app.cost.models import RunRecord, UsageProfile
from app.cost.pricing import get_pricing, models_by_provider
from app.cost.service import (
    DEFAULT_BASELINE_TOKENS,
    model_projection,
    revenue_projection,
    usage_profile,
)


class _StubRepo:
    def __init__(self, runs=()):
        self._runs = list(runs)

    async def list_all(self, agent_id, limit=100000):
        return list(self._runs)


def _run(baseline=4000, optimized=1000, output=300, model="gpt-4o-mini"):
    return RunRecord(
        agent_id="a",
        conversation_id="c",
        query="q",
        model=model,
        baseline_tokens=baseline,
        optimized_tokens=optimized,
        output_tokens=output,
        baseline_cost=0.0,
        optimized_cost=0.0,
        savings=0.0,
    )


def _profile(baseline=4000, optimized=1000, output=300):
    return UsageProfile(
        baseline_input_tokens=baseline,
        optimized_input_tokens=optimized,
        output_tokens=output,
        measured_runs=1,
        is_measured=True,
        data_source_note="",
    )


@pytest.mark.asyncio
async def test_usage_profile_is_measured_from_recorded_runs():
    repo = _StubRepo([_run(baseline=4000, optimized=1000), _run(baseline=2000, optimized=500)])
    profile = await usage_profile(repo, "a")
    assert profile.is_measured
    assert profile.measured_runs == 2
    assert profile.baseline_input_tokens == 3000  # real mean, not a constant
    assert profile.optimized_input_tokens == 750
    assert "Measured across 2" in profile.data_source_note


@pytest.mark.asyncio
async def test_usage_profile_falls_back_and_says_so():
    profile = await usage_profile(_StubRepo(), "a")
    assert profile.is_measured is False
    assert profile.baseline_input_tokens == DEFAULT_BASELINE_TOKENS
    assert "No recorded runs yet" in profile.data_source_note


def test_model_projection_math_matches_published_rates():
    """Hand-checked: 4000+1000 in, 300 out, 1 run, at mini pricing."""
    row = model_projection("gpt-4o-mini", _profile(), {"weekly": 1, "monthly": 1, "annual": 1})
    # gross: (4 * 0.15) + (0.3 * 0.60) = 0.60 + 0.18 = 0.78
    assert row.weekly_gross == 0.78
    # with MEMTRACE: (1 * 0.15) + 0.18 = 0.33
    assert row.weekly_with_memtrace == 0.33
    assert row.weekly_savings == 0.45
    assert row.savings_pct == 57.69




@pytest.mark.asyncio
async def test_deltas_are_relative_to_the_incumbent_model():
    repo = _StubRepo([_run()])
    proj = await revenue_projection(
        repository=repo,
        agent_id="a",
        agents=10,
        runs_per_agent_per_day=10,
        current_model="gpt-4o",
        compare_models=["gpt-4o-mini", "claude-3-5-sonnet"],
    )
    by_model = {r.model: r for r in proj.rows}
    # At identical measured token volumes: mini is cheaper than 4o, Sonnet
    # ($3/$15) is dearer than 4o ($2.50/$10). Both directions must show up,
    # otherwise the delta column is just decoration.
    assert by_model["gpt-4o-mini"].annual_delta_vs_current < 0
    assert by_model["claude-3-5-sonnet"].annual_delta_vs_current > 0
    assert by_model["gpt-4o"].annual_delta_vs_current == 0.0  # zero by definition
    assert proj.best_model == "gpt-4o-mini"
    assert proj.best_model_annual_savings > 0


@pytest.mark.asyncio
async def test_fleet_size_scales_the_numbers_exactly():
    repo = _StubRepo([_run()])
    small = await revenue_projection(repo, "a", agents=1, runs_per_agent_per_day=100, current_model="gpt-4o")
    big = await revenue_projection(repo, "a", agents=10, runs_per_agent_per_day=100, current_model="gpt-4o")
    assert big.runs_per_day == small.runs_per_day * 10
    assert big.rows[0].monthly_gross == round(small.rows[0].monthly_gross * 10, 2)


@pytest.mark.asyncio
async def test_horizon_run_counts_and_totals_are_consistent():
    repo = _StubRepo([_run()])
    proj = await revenue_projection(repo, "a", agents=2, runs_per_agent_per_day=50, current_model="gpt-4o")
    assert proj.weekly_runs == 2 * 50 * 7
    assert proj.monthly_runs == 2 * 50 * 30
    assert proj.annual_runs == 2 * 50 * 365
    row = proj.rows[0]
    assert row.annual_gross == round(row.weekly_gross * 365 / 7, 2)
    assert row.annual_savings == round(row.annual_gross - row.annual_with_memtrace, 2)


@pytest.mark.asyncio
async def test_no_fabricated_winner_when_incumbent_is_already_cheapest():
    """Reporting a 'best model' that costs more would be dishonest, so the
    incumbent is returned as the answer instead."""
    repo = _StubRepo([_run()])
    proj = await revenue_projection(
        repo, "a", agents=5, runs_per_agent_per_day=50,
        current_model="openai/gpt-oss-120b", compare_models=["claude-3-opus"],
    )
    assert proj.best_model is None
    assert proj.best_model_annual_savings == 0.0


@pytest.mark.asyncio
async def test_margin_points_only_reported_when_revenue_is_known():
    repo = _StubRepo([_run()])
    without = await revenue_projection(
        repo, "a", agents=10, runs_per_agent_per_day=50,
        current_model="gpt-4o", compare_models=["grok-3-mini"], revenue_per_agent_month=0.0,
    )
    assert without.margin_points_recovered == 0.0
    assert without.current_model_annual_spend_pct_of_revenue == 0.0

    with_rev = await revenue_projection(
        repo, "a", agents=10, runs_per_agent_per_day=50,
        current_model="gpt-4o", compare_models=["grok-3-mini"], revenue_per_agent_month=1000.0,
    )
    assert with_rev.annual_revenue == 120000.0
    assert with_rev.margin_points_recovered > 0


@pytest.mark.asyncio
async def test_unknown_model_still_appears_and_is_flagged():
    repo = _StubRepo([_run()])
    proj = await revenue_projection(
        repo, "a", agents=1, runs_per_agent_per_day=10,
        current_model="gpt-4o", compare_models=["some-unreleased-model"],
    )
    row = next(r for r in proj.rows if r.model == "some-unreleased-model")
    assert row.provider == "unknown"
    assert row.label == "Unpriced model"


def test_model_catalog_covers_every_requested_provider():
    grouped = models_by_provider()
    for provider in ("openai", "anthropic", "grok", "groq"):
        assert grouped[provider], f"expected at least one {provider} model"
    names = {m["model"] for models in grouped.values() for m in models}
    assert {"gpt-4o", "claude-3-5-sonnet", "grok-3", "openai/gpt-oss-120b"} <= names

def test_projection_scales_linearly_with_horizons():
    row = model_projection("gpt-4o-mini", _profile(), {"weekly": 7, "monthly": 30, "annual": 365})
    assert row.annual_gross == round(row.weekly_gross * 365 / 7, 2)


def test_free_tier_model_projects_to_zero_not_the_fallback_rate():
    """A ":free" model must not silently inherit gpt-4o-mini pricing — that
    would overstate savings, the exact kind of flattering number this view is
    meant to avoid."""
    row = model_projection("qwen/qwen3.8-27b:free", _profile(), {"weekly": 1, "monthly": 1, "annual": 1})
    assert row.weekly_gross == 0.0
    assert row.weekly_savings == 0.0
    assert row.savings_pct == 0.0


def test_output_tokens_are_billed_even_with_memory_filtering():
    """MEMTRACE shrinks input context only; claiming it shrinks output too
    would overstate the savings."""
    row = model_projection("gpt-4o-mini", _profile(), {"weekly": 1})
    p = get_pricing("gpt-4o-mini")
    assert row.weekly_gross - row.weekly_with_memtrace == pytest.approx(
        (4000 - 1000) / 1000 * p.input_per_1k, abs=1e-6
    )
