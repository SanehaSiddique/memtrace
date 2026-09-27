"""Thin OpenAI-compatible client built on httpx, so any compatible provider
(OpenAI, Azure OpenAI, local vLLM/Ollama gateway, etc.) works by swapping
OPENAI_BASE_URL. No official `openai` SDK dependency required.
"""

import logging
import time
from typing import List

import httpx

from app.llm.interface import BaseLLMClient

logger = logging.getLogger("uvicorn.error")


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

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        started = time.perf_counter()
        logger.info("[memtrace.llm] chat.start provider=openai model=%s", self._model)
        try:
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
                answer = data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "[memtrace.llm] chat.error provider=openai model=%s status=%s elapsed_ms=%.1f",
                self._model,
                exc.response.status_code,
                (time.perf_counter() - started) * 1000,
            )
            raise
        except Exception:
            logger.exception(
                "[memtrace.llm] chat.error provider=openai model=%s elapsed_ms=%.1f",
                self._model,
                (time.perf_counter() - started) * 1000,
            )
            raise

        logger.info(
            "[memtrace.llm] chat.complete provider=openai model=%s status=%s elapsed_ms=%.1f output_chars=%s",
            self._model,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            len(answer),
        )
        return answer

    async def embed(self, text: str) -> List[float]:
        started = time.perf_counter()
        logger.info("[memtrace.llm] embed.start provider=openai model=%s", self._embedding_model)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers=self._headers(),
                    json={"model": self._embedding_model, "input": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["data"][0]["embedding"]
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "[memtrace.llm] embed.error provider=openai model=%s status=%s elapsed_ms=%.1f",
                self._embedding_model,
                exc.response.status_code,
                (time.perf_counter() - started) * 1000,
            )
            raise
        except Exception:
            logger.exception(
                "[memtrace.llm] embed.error provider=openai model=%s elapsed_ms=%.1f",
                self._embedding_model,
                (time.perf_counter() - started) * 1000,
            )
            raise

        logger.info(
            "[memtrace.llm] embed.complete provider=openai model=%s status=%s elapsed_ms=%.1f dimensions=%s",
            self._embedding_model,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            len(embedding),
        )
        return embedding
