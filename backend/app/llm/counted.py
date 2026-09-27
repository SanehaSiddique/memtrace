"""Counting + caching wrapper around any `BaseLLMClient`.

Two problems this exists to fix, both visible in the OpenRouter 429 storm:

  1. The app had no idea how many provider requests a single user turn cost. One
     logical "answer this" could quietly become four HTTP attempts (the model
     fallback loop), and asking the same question twice paid for it twice. Every
     call now goes through here, so `/llm/stats` and each turn's `llm` block
     report *logical calls*, *cache hits*, *provider calls* (real HTTP requests),
     tokens and latency as measured facts instead of estimates — this is the
     "LLM calls" column the two-agent comparison needs.
  2. Deterministic repeats should be free. An identical (system, user,
     temperature) request is served from an in-process TTL cache without
     touching the network at all.

The wrapper is transparent: `is_live`, `embeddings_are_local`, `provider` and
`model_name` delegate to the wrapped client, so nothing else in the app has to
know it is there.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from app.llm.cache import DEFAULT_MAX_ENTRIES, DEFAULT_TTL_SECONDS, TTLCache, make_key
from app.llm.interface import BaseLLMClient, ChatCompletion

_COUNTER_KEYS = (
    "chat_calls",
    "chat_cache_hits",
    "chat_provider_calls",
    "chat_errors",
    "embed_calls",
    "embed_cache_hits",
    "embed_provider_calls",
    "rate_limit_events",
    "tokens_in",
    "tokens_out",
)


def _as_int(usage: dict, *keys: str) -> int:
    for key in keys:
        value = (usage or {}).get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


class CountedLLMClient(BaseLLMClient):
    """Same interface as the client it wraps, plus accounting and caching."""

    def __init__(
        self,
        inner: BaseLLMClient,
        cache_enabled: bool = True,
        chat_cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        chat_cache_max_entries: int = DEFAULT_MAX_ENTRIES,
        embed_cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        embed_cache_max_entries: int = 4096,
    ) -> None:
        self._inner = inner
        self._cache_enabled = cache_enabled
        self._chat_cache = TTLCache(ttl_seconds=chat_cache_ttl_seconds, max_entries=chat_cache_max_entries)
        self._embed_cache = TTLCache(ttl_seconds=embed_cache_ttl_seconds, max_entries=embed_cache_max_entries)
        self._counters: Dict[str, Any] = {key: 0 for key in _COUNTER_KEYS}
        self._counters["models_used"] = {}
        self._counters["latency_ms_total"] = 0.0
        self._counters["latency_ms_max"] = 0.0

    # -- identity passthrough --------------------------------------------------

    @property
    def is_live(self) -> bool:  # type: ignore[override]
        return self._inner.is_live

    @property
    def embeddings_are_local(self) -> bool:  # type: ignore[override]
        return self._inner.embeddings_are_local

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._inner.model_name

    @property
    def provider(self) -> str:  # type: ignore[override]
        return getattr(self._inner, "provider", type(self._inner).__name__)

    @property
    def inner(self) -> BaseLLMClient:
        """The wrapped client, for provider-specific probes (e.g. `/key` quota)."""
        return self._inner

    # -- chat ------------------------------------------------------------------

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        text, _usage, _model = await self.chat_with_usage(system, user, temperature)
        return text

    async def chat_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Tuple[str, dict, str]:
        key = make_key(self.model_name, "chat", f"{temperature}", system, user)
        if self._cache_enabled:
            hit, cached = self._chat_cache.get(key)
            if hit:
                self._counters["chat_cache_hits"] += 1
                return cached

        self._counters["chat_calls"] += 1
        provider_before = self._provider_call_count()
        started = time.monotonic()
        try:
            result = await self._inner.chat_with_usage(system, user, temperature)
        except Exception:
            self._record_latency(started)
            self._record_provider_calls(provider_before)
            self._record_rate_limit_events()
            self._counters["chat_errors"] += 1
            raise

        self._record_latency(started)
        self._record_provider_calls(provider_before)
        self._record_rate_limit_events()

        text, usage, model = result
        if model:
            self._counters["models_used"][model] = self._counters["models_used"].get(model, 0) + 1
        self._counters["tokens_in"] += _as_int(usage, "prompt_tokens", "input_tokens")
        self._counters["tokens_out"] += _as_int(usage, "completion_tokens", "output_tokens")

        if self._cache_enabled:
            self._chat_cache.set(key, result)
        return result

    # -- tool-calling chat completions -------------------------------------------

    async def chat_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> ChatCompletion:
        """Pass the full chat-completions contract through to the wrapped provider.

        The agent-comparison demo drives both agents through `chat_completion`
        (it needs the raw `tool_calls` and the number of tool schemas actually
        sent), so this wrapper must forward it rather than inheriting
        `BaseLLMClient`'s always-raising default. Accounting mirrors
        `chat_with_usage`, and caching is keyed on the full request so an
        identical tool-selection turn is still free on a repeat.
        """
        import json

        key = make_key(
            self.model_name,
            "chat_completion",
            model or "auto",
            f"{temperature}",
            json.dumps(messages, sort_keys=True, default=str),
            json.dumps(tools or [], sort_keys=True, default=str),
        )
        if self._cache_enabled:
            hit, cached = self._chat_cache.get(key)
            if hit and isinstance(cached, ChatCompletion):
                self._counters["chat_cache_hits"] += 1
                return cached

        self._counters["chat_calls"] += 1
        provider_before = self._provider_call_count()
        started = time.monotonic()
        try:
            completion = await self._inner.chat_completion(
                messages, tools=tools, temperature=temperature, model=model
            )
        except Exception:
            self._record_latency(started)
            self._record_provider_calls(provider_before)
            self._record_rate_limit_events()
            self._counters["chat_errors"] += 1
            raise

        self._record_latency(started)
        self._record_provider_calls(provider_before)
        self._record_rate_limit_events()

        if completion.model:
            models_used = self._counters["models_used"]
            models_used[completion.model] = models_used.get(completion.model, 0) + 1
        self._counters["tokens_in"] += _as_int(completion.usage, "prompt_tokens", "input_tokens")
        self._counters["tokens_out"] += _as_int(completion.usage, "completion_tokens", "output_tokens")

        if self._cache_enabled:
            self._chat_cache.set(key, completion)
        return completion

    # -- embeddings ------------------------------------------------------------

    async def embed(self, text: str) -> List[float]:
        key = make_key(self.model_name, "embed", text)
        if self._cache_enabled:
            hit, cached = self._embed_cache.get(key)
            if hit:
                self._counters["embed_cache_hits"] += 1
                return cached

        self._counters["embed_calls"] += 1
        provider_before = self._provider_call_count()
        try:
            embedding = await self._inner.embed(text)
        except Exception:
            self._record_provider_calls(provider_before)
            raise
        self._record_provider_calls(provider_before)

        if self._cache_enabled:
            self._embed_cache.set(key, embedding)
        return embedding

    # -- accounting ------------------------------------------------------------

    def _provider_call_count(self) -> Optional[int]:
        value = getattr(self._inner, "provider_call_count", None)
        return value if isinstance(value, int) else None

    def _record_provider_calls(self, before: Optional[int]) -> None:
        if before is not None:
            after = self._provider_call_count()
            self._counters["chat_provider_calls"] += max(0, (after or 0) - before)
        elif not self.embeddings_are_local:
            # Wrapped client doesn't report attempts (e.g. a plain OpenAI-compatible
            # transport): count the single request it must have made.
            self._counters["chat_provider_calls"] += 1

    def _record_rate_limit_events(self) -> None:
        value = getattr(self._inner, "rate_limit_events", None)
        if isinstance(value, int):
            self._counters["rate_limit_events"] = value

    def _record_latency(self, started: float) -> None:
        elapsed_ms = (time.monotonic() - started) * 1000
        self._counters["latency_ms_total"] += elapsed_ms
        self._counters["latency_ms_max"] = max(self._counters["latency_ms_max"], elapsed_ms)

    # -- reporting -------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Cumulative, process-lifetime accounting (JSON-safe)."""
        data: Dict[str, Any] = {key: self._counters[key] for key in _COUNTER_KEYS}
        data["models_used"] = dict(self._counters["models_used"])
        data["latency_ms_total"] = round(self._counters["latency_ms_total"], 2)
        data["latency_ms_max"] = round(self._counters["latency_ms_max"], 2)
        data["provider"] = self.provider
        data["model"] = self.model_name
        data["is_live"] = self.is_live
        data["embed_is_local"] = self.embeddings_are_local
        data["cache_enabled"] = self._cache_enabled
        data["chat_cache"] = self._chat_cache.stats()
        data["embed_cache"] = self._embed_cache.stats()
        data["accounting_note"] = (
            "chat_calls counts logical LLM calls requested by the app; chat_provider_calls counts real HTTP "
            "requests to the provider (rate-limit fallbacks add attempts); cache hits cost neither."
        )
        return data

    def delta(self, before: Dict[str, Any]) -> Dict[str, Any]:
        """What a single request cost: `after.delta(before)` for one turn."""
        data = {key: self._counters[key] - int(before.get(key, 0)) for key in _COUNTER_KEYS}
        data["latency_ms_total"] = round(
            self._counters["latency_ms_total"] - float(before.get("latency_ms_total", 0.0)), 2
        )
        previous_models: Dict[str, int] = before.get("models_used") or {}
        data["models_used"] = {
            model: count - previous_models.get(model, 0)
            for model, count in self._counters["models_used"].items()
            if count - previous_models.get(model, 0) > 0
        }
        return data

    def reset(self) -> None:
        for key in _COUNTER_KEYS:
            self._counters[key] = 0
        self._counters["models_used"] = {}
        self._counters["latency_ms_total"] = 0.0
        self._counters["latency_ms_max"] = 0.0
        self._chat_cache.clear()
        self._embed_cache.clear()
        self._chat_cache.reset_counters()
        self._embed_cache.reset_counters()
        inner_calls = getattr(self._inner, "provider_call_count", None)
        if isinstance(inner_calls, int):
            self._inner.provider_call_count = 0
