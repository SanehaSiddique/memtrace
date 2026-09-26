"""Deterministic, offline-safe LLM stand-in.

Used automatically whenever no OPENAI_API_KEY is configured, so the whole
pipeline (extraction, judgment fallback, answer generation) stays runnable
without network access or a paid key. `embed()` uses the hashing trick so
semantically similar text still lands closer in vector space than unrelated
text, which is enough to demonstrate hybrid retrieval honestly.
"""

import hashlib
import math
import re
from typing import List

from app.llm.interface import BaseLLMClient

EMBEDDING_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class MockLLMClient(BaseLLMClient):
    is_live = False

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError(
            "MockLLMClient cannot generate free-form text; callers must provide a "
            "deterministic fallback (see agent/nodes.py, memory/extraction.py)."
        )

    async def embed(self, text: str) -> List[float]:
        vector = [0.0] * EMBEDDING_DIM
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]
