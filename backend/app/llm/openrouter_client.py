"""OpenRouter chat client.

OpenRouter (https://openrouter.ai) is OpenAI-compatible for chat completions
and fronts a rotating catalog of free ":free"-suffixed models, which is what
makes it useful for running MEMTRACE with real answer generation and no
spend. It has no free embedding model, though, so this client reuses the
same offline hashing embedding as MockLLMClient rather than calling a paid
OpenRouter embeddings endpoint.
"""

from typing import List, Tuple

import httpx

from app.llm.hashing import hash_embed
from app.llm.interface import BaseLLMClient

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Other free models confirmed live during development — tried in order after the
# configured primary on a 429, since OpenRouter's free-tier congestion is
# per-model (a specific model's shared upstream pool filling up), not account-wide.
FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "liquid/lfm-2.5-2.6b:free",
    "cohere/north-mini-code:free",
]


class OpenRouterLLMClient(BaseLLMClient):
    is_live = True

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model
        self._site_url = site_url
        self._app_name = app_name

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model

    def _headers(self) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        # Optional attribution headers OpenRouter uses for its public leaderboard;
        # https://openrouter.ai/docs#headers. Harmless to omit.
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        return headers

    def _candidate_models(self) -> List[str]:
        others = [m for m in FALLBACK_MODELS if m != self._model]
        return [self._model, *others]

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        text, _usage, _model = await self._chat_with_usage(system, user, temperature)
        return text

    async def chat_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Tuple[str, dict, str]:
        """Like `chat()`, but also returns real token usage and the model that
        actually answered (may differ from the configured one after a
        fallback) — for honest per-message cost/token tracing."""
        return await self._chat_with_usage(system, user, temperature)

    async def _chat_with_usage(self, system: str, user: str, temperature: float) -> Tuple[str, dict, str]:
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            for model in self._candidate_models():
                try:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=self._headers(),
                        json={
                            "model": model,
                            "temperature": temperature,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    # OpenRouter sometimes returns HTTP 200 with an `error` body
                    # (no free capacity, moderation, context length, ...) instead
                    # of an HTTP error status — treat it like a 429 and fall
                    # through to the next candidate model.
                    if "error" in data:
                        last_error = RuntimeError(f"OpenRouter error for {model}: {data['error']}")
                        continue
                    return data["choices"][0]["message"]["content"], data.get("usage", {}), model
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code == 429:
                        continue  # this model's shared pool is congested — try the next one
                    raise
        raise last_error  # every candidate model was rate-limited or errored

    async def embed(self, text: str) -> List[float]:
        return hash_embed(text)
