"""OpenRouter chat client.

OpenRouter (https://openrouter.ai) is OpenAI-compatible for chat completions
and fronts a rotating catalog of free ":free"-suffixed models, which is what
makes it useful for running MEMTRACE with real answer generation and no
spend. It has no free embedding model, though, so this client reuses the
same offline hashing embedding as MockLLMClient rather than calling a paid
OpenRouter embeddings endpoint.

Rate limiting, honestly: free-model limits are enforced **account-wide, not per
key** — "making additional accounts or API keys will not affect your rate
limits, as we govern capacity globally" (openrouter.ai/docs/api-reference/limits)
— and they are per-minute *and* per-day, tiered by all-time credits purchased.
That is exactly why this app was dying with 429s: every logical LLM call
re-probed four candidate models in a tight loop, with no backoff, no memory of
which models had just rejected us, and no cache, so a handful of user turns
became dozens of requests against an already-congested shared pool.

This client now behaves like a well-mannered free-tier citizen:

  * a model that returns 429 goes into a cooldown and is *skipped* (not
    re-probed) on subsequent calls, and the last model that actually answered is
    tried first, so a healthy model keeps serving traffic;
  * 429s honour `Retry-After`, otherwise they back off exponentially;
  * after a 429 the client spaces requests adaptively for a minute (free-tier
    per-minute ceiling) so it doesn't immediately re-trigger the limit, then
    relaxes back to unpadded latency;
  * when every candidate is cooling down it fails fast with an actionable
    `LLMRateLimitedError` instead of firing more doomed requests;
  * `key_status()` reads GET /api/v1/key for the real remaining free-model quota.
"""

import asyncio
import time
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx

from app.llm.errors import LLMRateLimitedError
from app.llm.hashing import hash_embed
from app.llm.interface import BaseLLMClient, ChatCompletion
from app.llm.wire import (
    StreamRateLimited,
    parse_chat_completion,
    seconds_from_retry_after,
    stream_chat_completion,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0

# Other free models confirmed live during development — tried in order after the
# configured primary on a 429, since OpenRouter's free-tier congestion is
# per-model (a specific model's shared upstream pool filling up), not account-wide.
FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "liquid/lfm-2.5-2.6b:free",
    "cohere/north-mini-code:free",
]

DEFAULT_MAX_RPM = 20  # documented free-model per-minute ceiling
DEFAULT_MAX_BACKOFF_SECONDS = 20.0
DEFAULT_MODEL_COOLDOWN_SECONDS = 90.0
DEFAULT_BASE_BACKOFF_SECONDS = 1.0
_ADAPTIVE_WINDOW_SECONDS = 60.0  # how long we stay polite after a 429
_KEY_STATUS_TTL_SECONDS = 60.0


class _AttemptRateLimited(Exception):
    """Internal: a single attempt hit a 429 (HTTP status or a 200 error body)."""

    def __init__(self, detail: str, retry_after: Optional[float] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


# --- process-wide congestion state ------------------------------------------
# Shared by every instance, because the quota being spent belongs to the
# OpenRouter *account*, not to a client object — so cooldowns are account-level.
_model_cooldown_until: dict = {}
_last_good_model: Optional[str] = None
_adaptive_interval_seconds: float = 0.0
_adaptive_until: float = 0.0
_next_request_at: float = 0.0
_rate_limit_events: int = 0
_last_rate_limit: Optional[dict] = None
_key_status_cache: Tuple[float, Optional[dict]] = (0.0, None)


def _earliest_cooldown_seconds() -> Optional[float]:
    now = time.monotonic()
    remaining = [max(0.0, until - now) for until in _model_cooldown_until.values() if until > now]
    return round(min(remaining), 1) if remaining else None


def _rate_limit_message(retry_after: Optional[float]) -> str:
    wait = f"Retry in ~{retry_after:.0f}s" if retry_after else "Retry shortly"
    return (
        "Every candidate free model on this OpenRouter account is rate-limited/congested. "
        f"{wait}. OpenRouter's free-model limits are per-account, not per-key (extra keys/accounts do not "
        "raise them): point OPENROUTER_MODEL at a paid variant, purchase at least $10 of credits to raise "
        "the free daily ceiling, or wait for the shared pool to drain. Cached and deterministic paths in "
        "MEMTRACE keep working while the model is unavailable."
    )


class OpenRouterLLMClient(BaseLLMClient):
    is_live = True
    embeddings_are_local = True  # hashing embedder: no provider request, no quota
    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        max_rpm: int = DEFAULT_MAX_RPM,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        model_cooldown_seconds: float = DEFAULT_MODEL_COOLDOWN_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model
        self.model_name = model
        self._site_url = site_url
        self._app_name = app_name
        self._max_rpm = max(1, int(max_rpm))
        self._max_backoff_seconds = max(0.0, float(max_backoff_seconds))
        self._model_cooldown_seconds = max(0.0, float(model_cooldown_seconds))
        self._timeout = timeout
        self._transport = transport
        # Real HTTP attempts made (1..len(candidates) per logical call) and 429s
        # seen — `CountedLLMClient` turns these into honest per-turn metrics.
        self.provider_call_count = 0
        self.rate_limit_events = 0

    def _headers(self) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        # Optional attribution headers OpenRouter uses for its public leaderboard;
        # https://openrouter.ai/docs#headers. Harmless to omit.
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        return headers

    def _ordered_candidates(self) -> List[str]:
        """Last-known-good model first, then the configured model and fallbacks,
        minus anything currently in a 429 cooldown (no more re-probing a model
        that just rejected us on every single call)."""
        global _last_good_model
        ordered: List[str] = []
        if _last_good_model:
            ordered.append(_last_good_model)
        for candidate in [self._model, *FALLBACK_MODELS]:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        now = time.monotonic()
        return [m for m in ordered if _model_cooldown_until.get(m, 0.0) <= now]

    def _current_min_interval(self) -> float:
        """Adaptive spacing: only enforced for a short window after a 429, so
        normal (uncongested) latency is never artificially padded."""
        if time.monotonic() >= _adaptive_until:
            return 0.0
        return _adaptive_interval_seconds or (60.0 / self._max_rpm)

    async def _wait_for_slot(self) -> None:
        global _next_request_at
        interval = self._current_min_interval()
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now < _next_request_at:
            await asyncio.sleep(_next_request_at - now)
        _next_request_at = time.monotonic() + interval

    async def _backoff(self, index: int, retry_after: Optional[float]) -> None:
        delay = retry_after if retry_after is not None else DEFAULT_BASE_BACKOFF_SECONDS * (2**index)
        delay = min(max(delay, 0.0), self._max_backoff_seconds)
        if delay > 0:
            await asyncio.sleep(delay)

    def _note_rate_limit(self, model: str, exc: _AttemptRateLimited) -> None:
        global _adaptive_interval_seconds, _adaptive_until, _rate_limit_events, _last_rate_limit
        _model_cooldown_until[model] = time.monotonic() + self._model_cooldown_seconds
        self.rate_limit_events += 1
        _rate_limit_events += 1
        _adaptive_interval_seconds = max(_adaptive_interval_seconds, 60.0 / self._max_rpm)
        _adaptive_until = time.monotonic() + _ADAPTIVE_WINDOW_SECONDS
        _last_rate_limit = {"at": time.time(), "model": model, "detail": exc.detail}

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        text, _usage, _model = await self._chat_with_usage(system, user, temperature)
        return text

    async def chat_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Tuple[str, dict, str]:
        """Like `chat()`, but also returns real token usage and the model that
        actually answered (may differ from the configured one after a
        fallback) — for honest per-message cost/token tracing."""
        return await self._chat_with_usage(system, user, temperature)

    async def _chat_with_usage(self, system: str, user: str, temperature: float) -> Tuple[str, dict, str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        completion = await self._completion_with_fallback(
            lambda model: self._post_chat_completion(model, messages, None, temperature)
        )
        return completion.content, completion.usage, completion.model

    async def chat_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> ChatCompletion:
        """OpenAI-style chat completion with optional tool schemas — what the
        agent-comparison demo calls (app/core/llm_client.py). Passing `model`
        pins one candidate; otherwise the same last-known-good/fallback
        ordering (and 429 cooldown behavior) as `chat()` applies."""
        candidates = [model] if model else None
        return await self._completion_with_fallback(
            lambda candidate: self._post_chat_completion(candidate, messages, tools, temperature),
            candidates=candidates,
        )

    async def _completion_with_fallback(self, attempt, candidates: Optional[List[str]] = None) -> ChatCompletion:
        """Shared candidate-model loop: try the ordered candidates, put a 429ed
        model in cooldown, back off, and fail fast when every candidate is
        cooling down. Both the plain-text and tool-calling paths use this, so
        there is exactly one place that knows about free-tier rate limits."""
        global _last_good_model

        candidates = candidates or self._ordered_candidates()
        if not candidates:
            retry_after = _earliest_cooldown_seconds()
            raise LLMRateLimitedError(_rate_limit_message(retry_after), retry_after)

        last_detail = "no candidate model was attempted"
        for index, model in enumerate(candidates):
            await self._wait_for_slot()
            try:
                completion = await attempt(model)
            except _AttemptRateLimited as exc:
                self._note_rate_limit(model, exc)
                last_detail = exc.detail
                if index < len(candidates) - 1:
                    await self._backoff(index, exc.retry_after)
                continue
            _last_good_model = model
            return completion

        retry_after = _earliest_cooldown_seconds()
        raise LLMRateLimitedError(f"{last_detail}. {_rate_limit_message(retry_after)}", retry_after)

    async def _post_chat_completion(
        self, model: str, messages: List[dict], tools: Optional[List[dict]], temperature: float
    ) -> ChatCompletion:
        self.provider_call_count += 1
        payload: dict = {"model": model, "temperature": temperature, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )

        if response.status_code == 429:
            raise _AttemptRateLimited(
                f"{model} is rate-limited (HTTP 429)",
                seconds_from_retry_after(response.headers.get("Retry-After")),
            )
        response.raise_for_status()
        data = response.json()
        # OpenRouter sometimes returns HTTP 200 with an `error` body (no free
        # capacity, moderation, context length, ...) instead of an HTTP error
        # status — treat a rate/capacity error like a 429 and move on, but let
        # anything else (auth, malformed request) surface as a hard error.
        if "error" in data:
            error = data["error"] or {}
            message = str(error.get("message", ""))
            if error.get("code") == 429 or "rate" in message.lower() or "capacity" in message.lower():
                raise _AttemptRateLimited(f"{model} is congested ({error})")
            raise RuntimeError(f"OpenRouter error for {model}: {error}")
        return parse_chat_completion(data, fallback_model=model)

    async def chat_completion_stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> ChatCompletion:
        """Streaming counterpart to `chat_completion` — same candidate-model
        ordering, cooldown, and backoff via `_completion_with_fallback`; only
        how each attempt is made (streamed vs one-shot) differs."""
        candidates = [model] if model else None

        async def _attempt(candidate: str) -> ChatCompletion:
            self.provider_call_count += 1
            payload: dict = {"model": candidate, "temperature": temperature, "messages": messages}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                try:
                    return await stream_chat_completion(
                        client, f"{self._base_url}/chat/completions", self._headers(), payload, candidate, on_token
                    )
                except StreamRateLimited as exc:
                    raise _AttemptRateLimited(
                        f"{candidate} is rate-limited (HTTP 429)", exc.retry_after_seconds
                    ) from exc

        return await self._completion_with_fallback(_attempt, candidates=candidates)

    async def _post_chat(self, model: str, system: str, user: str, temperature: float) -> ChatCompletion:
        """Back-compat shim: the plain (system, user) call as a ChatCompletion."""
        return await self._post_chat_completion(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            None,
            temperature,
        )

    async def embed(self, text: str) -> List[float]:
        return hash_embed(text)

    async def key_status(self, max_age_seconds: float = _KEY_STATUS_TTL_SECONDS) -> Optional[dict]:
        """Real remaining free-model quota, straight from GET /api/v1/key.

        `free_model_daily_requests.used/limit/remaining` is the enforced daily
        counter for ":free" models; the per-minute ceiling isn't reported there.
        Cached briefly, and returns None (never raises) so stats endpoints stay up
        when the probe itself is unavailable.
        """
        global _key_status_cache
        cached_at, cached = _key_status_cache
        now = time.time()
        if cached is not None and now - cached_at < max_age_seconds:
            return cached
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.get(f"{self._base_url}/key", headers=self._headers())
                response.raise_for_status()
                data = (response.json() or {}).get("data") or {}
        except Exception:
            return None
        status = {
            "is_free_tier": data.get("is_free_tier"),
            "limit_remaining": data.get("limit_remaining"),
            "usage_daily": data.get("usage_daily"),
            "free_model_daily_requests": data.get("free_model_daily_requests"),
        }
        _key_status_cache = (now, status)
        return status

    def congestion_state(self) -> dict:
        """Current 429 telemetry, for `/llm/stats` — proof the client is backing
        off instead of hammering a dead model."""
        now = time.monotonic()
        return {
            "cooldowns_seconds": {
                model: round(max(0.0, until - now), 1)
                for model, until in _model_cooldown_until.items()
                if until > now
            },
            "last_good_model": _last_good_model,
            "rate_limit_events": _rate_limit_events,
            "last_rate_limit": _last_rate_limit,
            "adaptive_spacing_active": now < _adaptive_until,
            "adaptive_spacing_seconds": _adaptive_interval_seconds,
            "adaptive_window_remaining": round(max(0.0, _adaptive_until - now), 1),
            "max_rpm": self._max_rpm,
        }
