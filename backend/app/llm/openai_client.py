"""Thin OpenAI-compatible client built on httpx, so any compatible provider
(OpenAI, Azure OpenAI, local vLLM/Ollama gateway, etc.) works by swapping
OPENAI_BASE_URL. No official `openai` SDK dependency required.
"""

from typing import List

import httpx

from app.llm.interface import BaseLLMClient


class OpenAICompatibleLLMClient(BaseLLMClient):
    is_live = True

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._model = model
        self._embedding_model = embedding_model

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self._model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._embedding_model, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
