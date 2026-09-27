"""Thin OpenAI-compatible client built on httpx, so any compatible provider
(OpenAI, Azure OpenAI, local vLLM/Ollama gateway, etc.) works by swapping
OPENAI_BASE_URL. No official `openai` SDK dependency required.
"""

from typing import List, Optional, Tuple

import httpx

from app.llm.errors import LLMRateLimitedError
from app.llm.interface import BaseLLMClient, ChatCompletion
from app.llm.wire import parse_chat_completion


class OpenAICompatibleLLMClient(BaseLLMClient):
    is_live = True
    embeddings_are_local = False  # this one really calls /embeddings
    provider = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._model = model
        self.model_name = model
        self._embedding_model = embedding_model
        self._transport = transport
        # Mirrors OpenRouterLLMClient so the same counters work for both.
        self.provider_call_count = 0
        self.rate_limit_events = 0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        text, _usage, _model = await self.chat_with_usage(system, user, temperature)
        return text

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
        payload: dict = {"model": model or self._model, "temperature": temperature, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code == 429:
                self.rate_limit_events += 1
                raise LLMRateLimitedError(f"{payload['model']} is rate-limited (HTTP 429)")
            response.raise_for_status()
            data = response.json()
        return parse_chat_completion(data, fallback_model=str(payload["model"]))

    async def embed(self, text: str) -> List[float]:
        self.provider_call_count += 1
        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._embedding_model, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
