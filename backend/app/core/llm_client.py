"""The demo's single LLM client (docs/IMPLEMENTATION.md §4.1).

Contract: ``LLMClient.chat(messages, tools=None, model=...) -> LLMResponse``
where ``LLMResponse`` carries content, tool_calls, usage.prompt_tokens,
usage.completion_tokens, latency_ms, and the model that actually answered.

Everything else in the comparison demo (both agents, fact extraction) calls
*this* and never a provider directly, which is what makes the two agents
comparable: they share one client, one deterministic-response cache, and one
piece of accounting. Pieces of the doc that live here:

  * "selectable per request" provider choice (§4.1) — OpenRouter first (the
    configured free model), Groq as the A/B alternative, both real HTTP.
  * every call is a LangSmith span tagged agent_id/call_type (§4.1, §4.4).
  * call accounting the metrics collector reads back: logical calls, real
    provider requests, cache hits, tokens, and errors per call type.

Failures never masquerade as answers: when no provider can serve the request,
`LLMUnavailableError` propagates with the real provider detail, and the caller
records the failure in that turn's metrics.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.core.tracing import child_trace
from app.llm.cache import DEFAULT_MAX_ENTRIES, DEFAULT_TTL_SECONDS, TTLCache, make_key
from app.llm.errors import LLMRateLimitedError, LLMUnavailableError
from app.llm.interface import BaseLLMClient, ChatCompletion, ToolCall

# A provider hitting a transient 429 mid-turn must not silently blank out the
# answer (only Agent2's extra JEV round-trips make this common, since both
# agents fire concurrent requests at the same shared provider). Retry a couple
# times with backoff before treating the provider as exhausted for this call —
# OpenRouterLLMClient already does its own cooldown/backoff internally, so
# this mostly protects providers (e.g. Groq) that raise straight through.
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_BASE_BACKOFF_SECONDS = 1.5
_RATE_LIMIT_MAX_BACKOFF_SECONDS = 8.0


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_provider(cls, usage: Optional[dict]) -> "LLMUsage":
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


@dataclass
class LLMResponse:
    """Exactly the fields §4.1 requires, plus honest accounting extras."""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    cache_hit: bool = False
    provider_calls: int = 0

    @property
    def tool_call_names(self) -> List[str]:
        return [call.name for call in self.tool_calls]


def _cache_key(model: Optional[str], messages: List[dict], tools: Optional[List[dict]], temperature: float) -> str:
    import json

    return make_key(
        "llm.chat",
        model or "auto",
        f"{temperature:.4f}",
        json.dumps(messages, sort_keys=True, default=str),
        json.dumps(tools or [], sort_keys=True, default=str),
    )


class LLMClient:
    """One client, several real providers, one accounting ledger."""

    def __init__(
        self,
        providers: List[BaseLLMClient],
        cache_enabled: bool = True,
        cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        cache_max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._providers = [p for p in providers if p is not None]
        self._cache_enabled = cache_enabled
        self._cache = TTLCache(ttl_seconds=cache_ttl_seconds, max_entries=cache_max_entries)
        self._counters: Dict[str, Any] = {
            "logical_calls": 0,
            "provider_calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms_total": 0.0,
        }
        self._by_call_type: Dict[str, Dict[str, int]] = {}

    # -- identity ---------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return any(p.is_live for p in self._providers)

    @property
    def model_name(self) -> str:
        return ", ".join(p.model_name for p in self._providers) or "none"

    @property
    def providers(self) -> List[BaseLLMClient]:
        return list(self._providers)

    def provider_names(self) -> List[str]:
        return [getattr(p, "provider", type(p).__name__) for p in self._providers]

    # -- the contract -----------------------------------------------------------

    async def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        agent_id: str = "",
        call_type: str = "reasoning",
        run_group_id: str = "",
    ) -> LLMResponse:
        """One logical LLM call.

        `tools` are OpenAI-style tool schemas; the caller decides how many to
        send (Agent1 sends the whole registry, Agent2 sends at most one) — that
        difference is a headline metric, so this client counts it rather than
        hiding it.
        """
        started = time.perf_counter()
        self._counters["logical_calls"] += 1
        self._bump_call_type(call_type, "logical_calls")

        key = _cache_key(model, messages, tools, temperature)
        if self._cache_enabled:
            hit, cached = self._cache.get(key)
            if hit and isinstance(cached, LLMResponse):
                self._counters["cache_hits"] += 1
                self._bump_call_type(call_type, "cache_hits")
                return LLMResponse(**{**cached.__dict__, "cache_hit": True})

        errors: List[str] = []
        with child_trace(
            name=f"llm.{call_type}",
            run_type="llm",
            agent_id=agent_id,
            run_group_id=run_group_id,
            component="llm",
            call_type=call_type,
            metadata={
                "tools_in_prompt": len(tools or []),
                "tool_names": [(t.get("function") or {}).get("name") for t in (tools or [])],
                "messages": len(messages),
                "requested_model": model or "auto",
                "providers": self.provider_names(),
            },
        ) as run:
            for provider in self._providers:
                before = getattr(provider, "provider_call_count", None)
                try:
                    completion: ChatCompletion = await self._chat_completion_with_retry(
                        provider, messages, tools=tools, temperature=temperature, model=model
                    )
                except LLMUnavailableError as exc:
                    errors.append(f"{provider.provider}: {exc}")
                    continue
                except Exception as exc:  # provider bug/parse failure: try the next provider
                    errors.append(f"{provider.provider}: {type(exc).__name__}: {exc}")
                    continue

                response = LLMResponse(
                    content=completion.content,
                    tool_calls=list(completion.tool_calls),
                    usage=LLMUsage.from_provider(completion.usage),
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    model=completion.model or getattr(provider, "model_name", ""),
                    provider=getattr(provider, "provider", type(provider).__name__),
                    cache_hit=False,
                    provider_calls=self._provider_delta(provider, before),
                )
                self._record_success(response, call_type)
                if self._cache_enabled:
                    self._cache.set(key, response)
                try:
                    run.outputs = {
                        "content_chars": len(response.content),
                        "tool_calls": response.tool_call_names,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "model": response.model,
                        "provider": response.provider,
                    }
                except Exception:
                    pass  # tracing must never break a real call
                return response

            self._counters["errors"] += 1
            self._bump_call_type(call_type, "errors")
            detail = " | ".join(errors) or "no provider configured"
            raise LLMUnavailableError(f"No LLM provider could answer this call ({detail})")

    async def chat_stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        agent_id: str = "",
        call_type: str = "final_answer",
        run_group_id: str = "",
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> LLMResponse:
        """Streaming counterpart to `chat()` (docs/IMPLEMENTATION_V2.md §5.2
        `llm_final_answer_token`): calls `on_token` for each content delta as it
        arrives. Same cache, provider fallback, rate-limit retry, and accounting
        as `chat()` — streaming only changes how content is delivered while the
        call is in flight, never what gets counted afterward."""
        started = time.perf_counter()
        self._counters["logical_calls"] += 1
        self._bump_call_type(call_type, "logical_calls")

        key = _cache_key(model, messages, tools, temperature)
        if self._cache_enabled:
            hit, cached = self._cache.get(key)
            if hit and isinstance(cached, LLMResponse):
                self._counters["cache_hits"] += 1
                self._bump_call_type(call_type, "cache_hits")
                if on_token and cached.content:
                    await on_token(cached.content)
                return LLMResponse(**{**cached.__dict__, "cache_hit": True})

        errors: List[str] = []
        with child_trace(
            name=f"llm.{call_type}",
            run_type="llm",
            agent_id=agent_id,
            run_group_id=run_group_id,
            component="llm",
            call_type=call_type,
            metadata={
                "tools_in_prompt": len(tools or []),
                "tool_names": [(t.get("function") or {}).get("name") for t in (tools or [])],
                "messages": len(messages),
                "requested_model": model or "auto",
                "providers": self.provider_names(),
                "streaming": True,
            },
        ) as run:
            for provider in self._providers:
                before = getattr(provider, "provider_call_count", None)
                try:
                    completion: ChatCompletion = await self._chat_completion_stream_with_retry(
                        provider, messages, tools=tools, temperature=temperature, model=model, on_token=on_token
                    )
                except LLMUnavailableError as exc:
                    errors.append(f"{provider.provider}: {exc}")
                    continue
                except Exception as exc:  # provider bug/parse failure: try the next provider
                    errors.append(f"{provider.provider}: {type(exc).__name__}: {exc}")
                    continue

                response = LLMResponse(
                    content=completion.content,
                    tool_calls=list(completion.tool_calls),
                    usage=LLMUsage.from_provider(completion.usage),
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    model=completion.model or getattr(provider, "model_name", ""),
                    provider=getattr(provider, "provider", type(provider).__name__),
                    cache_hit=False,
                    provider_calls=self._provider_delta(provider, before),
                )
                self._record_success(response, call_type)
                if self._cache_enabled:
                    self._cache.set(key, response)
                try:
                    run.outputs = {
                        "content_chars": len(response.content),
                        "tool_calls": response.tool_call_names,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "model": response.model,
                        "provider": response.provider,
                    }
                except Exception:
                    pass  # tracing must never break a real call
                return response

            self._counters["errors"] += 1
            self._bump_call_type(call_type, "errors")
            detail = " | ".join(errors) or "no provider configured"
            raise LLMUnavailableError(f"No LLM provider could answer this call ({detail})")

    async def _chat_completion_with_retry(
        self,
        provider: BaseLLMClient,
        messages: List[dict],
        tools: Optional[List[dict]],
        temperature: float,
        model: Optional[str],
    ) -> ChatCompletion:
        """Retry the same provider on a transient rate limit before giving up on it."""
        return await self._with_rate_limit_retry(
            lambda: provider.chat_completion(messages, tools=tools, temperature=temperature, model=model)
        )

    async def _chat_completion_stream_with_retry(
        self,
        provider: BaseLLMClient,
        messages: List[dict],
        tools: Optional[List[dict]],
        temperature: float,
        model: Optional[str],
        on_token: Optional[Callable[[str], Awaitable[None]]],
    ) -> ChatCompletion:
        """Streaming counterpart to `_chat_completion_with_retry`."""
        return await self._with_rate_limit_retry(
            lambda: provider.chat_completion_stream(
                messages, tools=tools, temperature=temperature, model=model, on_token=on_token
            )
        )

    async def _with_rate_limit_retry(self, call: Callable[[], Awaitable[ChatCompletion]]) -> ChatCompletion:
        """Retry the same call on a transient rate limit before giving up on it."""
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return await call()
            except LLMRateLimitedError as exc:
                if attempt >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                backoff = exc.retry_after_seconds or (_RATE_LIMIT_BASE_BACKOFF_SECONDS * (attempt + 1))
                await asyncio.sleep(min(backoff, _RATE_LIMIT_MAX_BACKOFF_SECONDS))
        raise AssertionError("unreachable")  # loop always returns or raises

    # -- accounting -------------------------------------------------------------

    def _provider_delta(self, provider: BaseLLMClient, before: Optional[int]) -> int:
        if isinstance(before, int):
            after = getattr(provider, "provider_call_count", before)
            return max(0, int(after) - before)
        return 1 if provider.is_live else 0

    def _record_success(self, response: LLMResponse, call_type: str) -> None:
        self._counters["provider_calls"] += response.provider_calls
        self._counters["tokens_in"] += response.usage.prompt_tokens
        self._counters["tokens_out"] += response.usage.completion_tokens
        self._counters["latency_ms_total"] += response.latency_ms
        self._bump_call_type(call_type, "provider_calls", response.provider_calls)
        self._bump_call_type(call_type, "tokens_in", response.usage.prompt_tokens)
        self._bump_call_type(call_type, "tokens_out", response.usage.completion_tokens)

    def _bump_call_type(self, call_type: str, counter: str, amount: int = 1) -> None:
        bucket = self._by_call_type.setdefault(call_type, {})
        bucket[counter] = bucket.get(counter, 0) + amount

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats()

    def snapshot(self) -> Dict[str, Any]:
        return {
            **self._counters,
            "latency_ms_total": round(self._counters["latency_ms_total"], 2),
            "by_call_type": {k: dict(v) for k, v in self._by_call_type.items()},
            "providers": self.provider_names(),
            "models": self.model_name,
            "is_live": self.is_live,
            "cache": self._cache.stats(),
            "accounting_note": (
                "logical_calls = LLM calls the app asked for; provider_calls = real HTTP requests to the "
                "provider; cache_hits = identical requests served locally with zero provider traffic."
            ),
        }

    def delta(self, before: Dict[str, Any]) -> Dict[str, Any]:
        """What one turn cost, measured as after.delta(before)."""
        data = {key: self._counters[key] - int(before.get(key, 0)) for key in self._counters}
        data["latency_ms_total"] = round(
            self._counters["latency_ms_total"] - float(before.get("latency_ms_total", 0.0)), 2
        )
        previous = before.get("by_call_type") or {}
        data["by_call_type"] = {
            call_type: {
                counter: amount - (previous.get(call_type, {}) or {}).get(counter, 0)
                for counter, amount in bucket.items()
            }
            for call_type, bucket in self._by_call_type.items()
        }
        return data


