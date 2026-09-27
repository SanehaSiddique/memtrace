"""Deterministic, offline-safe LLM stand-in.

Used automatically whenever no OPENAI_API_KEY is configured, so the whole
pipeline (extraction, judgment fallback, answer generation) stays runnable
without network access or a paid key. `embed()` uses the hashing trick so
semantically similar text still lands closer in vector space than unrelated
text, which is enough to demonstrate hybrid retrieval honestly.
"""

from typing import List

from app.llm.hashing import hash_embed
from app.llm.interface import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    is_live = False
    embeddings_are_local = True
    provider = "mock"
    model_name = "mock-hashing-embedder"

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError(
            "MockLLMClient cannot generate free-form text; callers must provide a "
            "deterministic fallback (see agent/nodes.py, memory/extraction.py)."
        )

    async def embed(self, text: str) -> List[float]:
        return hash_embed(text)
