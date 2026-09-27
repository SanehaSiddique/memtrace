"""Cost projection for the agent-comparison demo (docs/IMPLEMENTATION.md §4.5).

The demo runs on free models, so the *actual* bill for the tokens a turn really
consumed is ~$0. That is not interesting. What is interesting is: "those same
real token counts, billed at the published rate of a paid model, would cost
this much" — which is what `project_cost()` computes.

Honesty rules baked in here:
  * ``project_cost`` is pure arithmetic over the *actual counted usage* returned
    by the provider for that turn. Token counts are never invented or scaled.
  * Every row carries the source URL and the date the rate was fetched, so the
    dashboard can say where a number came from instead of implying live billing.
  * Free models are priced at exactly $0 rather than falling back to a paid
    rate (see ``app.cost.pricing`` for the actual-cost side of the ledger).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from app.cost.pricing import ModelPricing, get_pricing

# The date these published rates were fetched at build time (the doc marks the
# exact models/prices as an OPEN DECISION to resolve at build time; these are
# the current published rates on that date).
PRICING_FETCHED_AT = "2026-09-27"


@dataclass(frozen=True)
class ProjectionPricing:
    """One published paid-model rate row, per 1M tokens (USD)."""

    key: str  # stable id used in the metrics payload (e.g. "gpt-4o")
    provider: str  # "openai" | "anthropic" | "xai"
    model_name: str  # the exact model id the rate belongs to
    label: str  # dashboard display name
    input_price_per_1m: float
    output_price_per_1m: float
    source_url: str
    notes: str = ""


# Published rates fetched 2026-09-27 from each provider's own pricing page.
PROJECTION_PRICING: Dict[str, ProjectionPricing] = {
    row.key: row
    for row in (
        ProjectionPricing(
            key="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            label="GPT-4o",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            source_url="https://developers.openai.com/api/docs/pricing",
        ),
        ProjectionPricing(
            key="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            label="GPT-4o mini",
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            source_url="https://developers.openai.com/api/docs/pricing",
        ),
        ProjectionPricing(
            key="claude-sonnet",
            provider="anthropic",
            model_name="claude-sonnet-5",
            label="Claude Sonnet 5",
            input_price_per_1m=2.00,
            output_price_per_1m=10.00,
            source_url="https://docs.anthropic.com/en/docs/about-claude/pricing",
        ),
        ProjectionPricing(
            key="claude-haiku",
            provider="anthropic",
            model_name="claude-haiku-4-5",
            label="Claude Haiku 4.5",
            input_price_per_1m=1.00,
            output_price_per_1m=5.00,
            source_url="https://docs.anthropic.com/en/docs/about-claude/pricing",
        ),
        ProjectionPricing(
            key="grok",
            provider="xai",
            model_name="grok-4.7",
            label="Grok 4.7",
            input_price_per_1m=2.00,
            output_price_per_1m=6.00,
            source_url="https://docs.x.ai/developers/pricing",
            notes="Short-context rate (<200k prompt tokens), global endpoint.",
        ),
    )
}

# The three headline models the CostComparison chart renders side by side, per
# docs/IMPLEMENTATION.md §7 ("gpt-4o", "claude-sonnet", "grok").
DEFAULT_PROJECTION_KEYS = ("gpt-4o", "claude-sonnet", "grok")


def project_cost(prompt_tokens: int, completion_tokens: int, pricing: ProjectionPricing) -> float:
    """Pure function: what this turn's *real* token counts would cost on a paid
    model. No LLM call, no estimation — just the published rate applied to the
    usage the provider actually reported."""
    return (
        max(0, int(prompt_tokens)) * pricing.input_price_per_1m
        + max(0, int(completion_tokens)) * pricing.output_price_per_1m
    ) / 1_000_000


def project_cost_all(
    prompt_tokens: int,
    completion_tokens: int,
    keys: Iterable[str] = DEFAULT_PROJECTION_KEYS,
) -> Dict[str, float]:
    """`{model_key: projected_usd}` for the headline models, rounded to a
    sub-cent precision that still shows real differences at demo scale."""
    return {
        key: round(project_cost(prompt_tokens, completion_tokens, PROJECTION_PRICING[key]), 8)
        for key in keys
        if key in PROJECTION_PRICING
    }


def actual_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Actual spend on the model that really ran (a free model is $0, honestly)."""
    pricing: ModelPricing = get_pricing(model)
    return (
        max(0, int(prompt_tokens)) * pricing.input_per_1k + max(0, int(completion_tokens)) * pricing.output_per_1k
    ) / 1000.0


def pricing_rows(keys: Optional[Iterable[str]] = None) -> List[ProjectionPricing]:
    keys = keys or PROJECTION_PRICING.keys()
    return [PROJECTION_PRICING[k] for k in keys if k in PROJECTION_PRICING]


def estimate_tokens(text: str) -> int:
    """Token estimate for payloads no provider ever counted for us (raw MCP tool
    JSON, filtered chunks). The doc explicitly asks for an *estimate* here
    (§6.2 step 7); it uses the standard ~4 characters/token approximation that
    Anthropic documents, and is labelled as an estimate in the metrics
    payload so the dashboard never presents it as provider-reported usage."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))
