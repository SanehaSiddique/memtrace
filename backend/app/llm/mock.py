"""Deterministic test double. Runtime provider factories never select it."""

import re
from typing import List

from app.llm.hashing import hash_embed
from app.llm.interface import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    is_live = False

    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        pattern = re.compile(
            r"^- \[[A-Z_]+\] .+? "
            r"(?:uses_database|deployed_on|uses_auth|depends_on|runs_on|rejected|considering|mentioned|selected|uses) "
            r"(?P<object>.+?) — (?P<content>.+?) \(confidence=",
            re.MULTILINE,
        )
        facts = []
        for match in pattern.finditer(user):
            reason = match.group("content").partition(" because ")[2].rstrip(".!")
            facts.append(match.group("object") + (f" because {reason}" if reason else ""))
        return ". ".join(facts) if facts else "I don't have a currently valid memory to answer that."

    async def embed(self, text: str) -> List[float]:
        return hash_embed(text)
