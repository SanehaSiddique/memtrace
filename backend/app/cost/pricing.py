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
    provider: str = "unknown"
    label: str = ""


# Approximate, published rates at time of writing. Demo/configured pricing —
# not pulled live from any provider. Every rate here flows through
# `get_pricing()`; nothing else in the app may hardcode a $/token number.
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # -- OpenAI ---------------------------------------------------------------
    "gpt-4o-mini": ModelPricing(input_per_1k=0.15, output_per_1k=0.60, provider="openai", label="GPT-4o mini"),
    "gpt-4o": ModelPricing(input_per_1k=2.50, output_per_1k=10.00, provider="openai", label="GPT-4o"),
    "gpt-4.1": ModelPricing(input_per_1k=2.00, output_per_1k=8.00, provider="openai", label="GPT-4.1"),
    "gpt-4.1-mini": ModelPricing(input_per_1k=0.40, output_per_1k=1.60, provider="openai", label="GPT-4.1 mini"),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.50, output_per_1k=1.50, provider="openai", label="GPT-3.5 Turbo"),
    # -- Anthropic ------------------------------------------------------------
    "claude-3-5-haiku": ModelPricing(input_per_1k=0.80, output_per_1k=4.00, provider="anthropic", label="Claude 3.5 Haiku"),
    "claude-3-5-sonnet": ModelPricing(input_per_1k=3.00, output_per_1k=15.00, provider="anthropic", label="Claude 3.5 Sonnet"),
    "claude-3-opus": ModelPricing(input_per_1k=15.00, output_per_1k=75.00, provider="anthropic", label="Claude 3 Opus"),
    # -- Grok (xAI) -----------------------------------------------------------
    "grok-3-mini": ModelPricing(input_per_1k=0.30, output_per_1k=0.50, provider="grok", label="Grok 3 mini"),
    "grok-3": ModelPricing(input_per_1k=3.00, output_per_1k=15.00, provider="grok", label="Grok 3"),
    "grok-4": ModelPricing(input_per_1k=3.00, output_per_1k=15.00, provider="grok", label="Grok 4"),
    # -- Groq (fast free/cheap hosted open models) ---------------------------
    "openai/gpt-oss-120b": ModelPricing(input_per_1k=0.10, output_per_1k=0.50, provider="groq", label="GPT-OSS 120B (Groq)"),
    "llama-3.3-70b-versatile": ModelPricing(input_per_1k=0.59, output_per_1k=0.79, provider="groq", label="Llama 3.3 70B (Groq)"),
    # -- OpenRouter ":free" models: actually $0/token — reflect that honestly
    #    instead of silently falling back to the gpt-4o-mini rate below.
    "nvidia/nemotron-3-super-120b-a12b:free": ModelPricing(input_per_1k=0, output_per_1k=0, provider="openrouter", label="Nemotron 3 Super (free)"),
    "nvidia/nemotron-3-ultra-550b-a55b:free": ModelPricing(input_per_1k=0, output_per_1k=0, provider="openrouter", label="Nemotron 3 Ultra (free)"),
    "liquid/lfm-2.5-2.6b:free": ModelPricing(input_per_1k=0, output_per_1k=0, provider="openrouter", label="LFM 2.5 2.6B (free)"),
    "cohere/north-mini-code:free": ModelPricing(input_per_1k=0, output_per_1k=0, provider="openrouter", label="North Mini Code (free)"),
    "google/gemma-4-31b-it:free": ModelPricing(input_per_1k=0, output_per_1k=0, provider="openrouter", label="Gemma 4 31B (free)"),
    "qwen/qwen3.8-27b:free": ModelPricing(input_per_1k=0, output_per_1k=0, provider="openrouter", label="Qwen 3.8 27B (free)"),
}
FALLBACK_PRICING = ModelPricing(input_per_1k=0.15, output_per_1k=0.60, provider="unknown", label="Unpriced model")

# Providers surfaced in the Executive "model mix" selector, in display order.
PROVIDER_ORDER = ["openai", "anthropic", "grok", "groq", "openrouter"]


def get_pricing(model: str) -> ModelPricing:
    return DEFAULT_PRICING.get(model, FALLBACK_PRICING)


def models_by_provider() -> Dict[str, list]:
    """The priced catalog grouped by provider, for the UI's model picker."""
    grouped: Dict[str, list] = {p: [] for p in PROVIDER_ORDER}
    for name, pricing in DEFAULT_PRICING.items():
        grouped.setdefault(pricing.provider, []).append(
            {"model": name, "label": pricing.label, "input_per_1k": pricing.input_per_1k, "output_per_1k": pricing.output_per_1k}
        )
    return grouped
