"""Keep the test suite hermetic and deterministic regardless of whatever
live provider keys happen to be configured in .env — tests exercise the
built-in fallback logic, not real network calls to a real LLM provider.
"""

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _no_live_llm_in_tests():
    keys = ["openai_api_key", "openrouter_api_key", "typesafe_api_key", "jev_api_key", "ai_gateway_api_key", "g8_api_key"]
    originals = {k: getattr(settings, k) for k in keys}
    for k in keys:
        setattr(settings, k, None)
    yield
    for k, v in originals.items():
        setattr(settings, k, v)
