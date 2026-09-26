"""LangSmith wiring.

MEMTRACE doesn't build its own tracing UI — LangSmith is the execution-trace
and evaluation layer, per the project brief. This module just makes sure it's
configured safely: tracing is force-disabled unless both the flag and an API
key are present, so the app never fails or hangs trying to phone home during
an offline demo. `traced` re-exports `langsmith.traceable`, already a no-op
wrapper when tracing is off, so nodes can use it unconditionally.
"""

import os

from langsmith import traceable as traced  # re-exported for app.agent.nodes

from app.config import Settings

__all__ = ["configure_langsmith", "traced"]


def configure_langsmith(settings: Settings) -> bool:
    enabled = settings.langchain_tracing_v2.strip().lower() == "true" and bool(settings.langchain_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    return enabled
