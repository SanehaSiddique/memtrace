"""Configurable model pricing.

Nothing else in the app should hardcode a $/token number — every cost
calculation flows through `get_pricing()`. Swap these for your provider's
actual published rates before treating the dashboard's dollar figures as
anything more than a configured projection.
"""

from typing import Dict

from pydantic import BaseModel


class ModelPricing(BaseModel):
    input_per_1k: float
    output_per_1k: float


# Approximate, published rates at time of writing. Demo/configured pricing —
# not pulled live from any provider.
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(input_per_1k=0.15, output_per_1k=0.60),
    "gpt-4o": ModelPricing(input_per_1k=2.50, output_per_1k=10.00),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.50, output_per_1k=1.50),
    # OpenRouter ":free" models actually are $0/token — reflect that honestly
    # instead of silently falling back to the gpt-4o-mini rate below.
    "nvidia/nemotron-3-super-120b-a12b:free": ModelPricing(input_per_1k=0, output_per_1k=0),
    "nvidia/nemotron-3-ultra-550b-a55b:free": ModelPricing(input_per_1k=0, output_per_1k=0),
    "liquid/lfm-2.5-2.6b:free": ModelPricing(input_per_1k=0, output_per_1k=0),
    "cohere/north-mini-code:free": ModelPricing(input_per_1k=0, output_per_1k=0),
    "google/gemma-4-31b-it:free": ModelPricing(input_per_1k=0, output_per_1k=0),
    "qwen/qwen3.8-27b:free": ModelPricing(input_per_1k=0, output_per_1k=0),
}
FALLBACK_PRICING = ModelPricing(input_per_1k=0.15, output_per_1k=0.60)


def get_pricing(model: str) -> ModelPricing:
    return DEFAULT_PRICING.get(model, FALLBACK_PRICING)
