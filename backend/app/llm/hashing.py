"""Deterministic, offline-safe text embedding via the hashing trick.

Shared by MockLLMClient and any live chat client (e.g. OpenRouter's free
tier) that has no free embedding model to call. Semantically similar text
still lands closer in vector space than unrelated text, which is enough to
demonstrate hybrid retrieval honestly without a network call.
"""

import hashlib
import math
import re
from typing import List

EMBEDDING_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def hash_embed(text: str) -> List[float]:
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
