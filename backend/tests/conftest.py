"""Keep tests hermetic by injecting explicit doubles at the API boundary."""

import pytest

from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient
from app.memory.repository import SQLiteMemoryRepository


@pytest.fixture(autouse=True)
def _no_live_llm_in_tests():
    keys = ["openai_api_key", "openrouter_api_key", "typesafe_api_key", "jev_api_key", "ai_gateway_api_key", "g8_api_key"]
    originals = {k: getattr(settings, k) for k in keys}
    for k in keys:
        setattr(settings, k, None)
    yield
    for k, v in originals.items():
        setattr(settings, k, v)
