"""Groq chat client.

Groq (https://groq.com) provides an OpenAI-compatible API for fast LLM inference.
Like OpenRouterLLMClient, it reuses local hashing embeddings so calls remain fast
and no external embedding quota is consumed.
"""

import asyncio
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx

from app.llm.errors import LLMRateLimitedError, LLMUnavailableError
from app.llm.hashing import hash_embed
from app.llm.interface import BaseLLMClient, ChatCompletion
from app.llm.wire import StreamRateLimited, parse_chat_completion, stream_chat_completion

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_SECONDS = 30.0


class GroqLLMClient(BaseLLMClient):
    is_live = True
    embeddings_are_local = True  # hashing embedder: no provider request, no quota
    provider = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        base_url: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._model = model.strip() or DEFAULT_GROQ_MODEL
        self.model_name = self._model
        self._base_url = (base_url or DEFAULT_GROQ_BASE_URL).rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport
        self.provider_call_count = 0
        self.rate_limit_events = 0

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        completion = await self.chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        return completion.content

    async def chat_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Tuple[str, dict, str]:
        completion = await self.chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        return completion.content, completion.usage, completion.model

    async def chat_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> ChatCompletion:
        self.provider_call_count += 1
        chosen_model = model or self._model
        payload: dict = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
            except httpx.RequestError as exc:
                raise LLMUnavailableError(f"Groq network error: {exc}") from exc

            if response.status_code == 429:
                self.rate_limit_events += 1
                raise LLMRateLimitedError(f"Groq model {chosen_model} is rate-limited (HTTP 429)")

            if response.status_code >= 400:
                raise LLMUnavailableError(f"Groq API error ({response.status_code}): {response.text}")

            data = response.json()

        return parse_chat_completion(data, fallback_model=chosen_model)

    async def chat_completion_stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> ChatCompletion:
        self.provider_call_count += 1
        chosen_model = model or self._model
        payload: dict = {"model": chosen_model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                return await stream_chat_completion(
                    client, f"{self._base_url}/chat/completions", self._headers(), payload, chosen_model, on_token
                )
            except StreamRateLimited as exc:
                self.rate_limit_events += 1
                raise LLMRateLimitedError(
                    f"Groq model {chosen_model} is rate-limited (HTTP 429)", exc.retry_after_seconds
                ) from exc
            except httpx.RequestError as exc:
                raise LLMUnavailableError(f"Groq network error: {exc}") from exc

    async def embed(self, text: str) -> List[float]:
        # Fast local hash embeddings (consistent dimension with Mock & OpenRouter)
        return hash_embed(text)

