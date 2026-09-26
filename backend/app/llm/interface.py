"""LLM client interface. Callers depend on this, never on a concrete provider."""

from abc import ABC, abstractmethod
from typing import List


class BaseLLMClient(ABC):
    """OpenAI-compatible chat + embeddings client."""

    is_live: bool = False

    @abstractmethod
    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str: ...

    @abstractmethod
    async def embed(self, text: str) -> List[float]: ...

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(t) for t in texts]
