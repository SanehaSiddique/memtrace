"""Keep tests hermetic by injecting explicit doubles at the API boundary."""

import pytest

from app.judgment.mock import MockMemoryJudge
from app.llm.mock import MockLLMClient


@pytest.fixture(autouse=True)
def _inject_test_doubles(monkeypatch):
    from app.api import container as container_module

    monkeypatch.setattr(container_module, "get_llm_client", lambda _settings: MockLLMClient())
    monkeypatch.setattr(container_module, "get_memory_judge", lambda _settings, _llm: MockMemoryJudge())
