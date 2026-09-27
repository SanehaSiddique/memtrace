"""Provider-neutral LLM failure types.

The agent layer (`app.agent.nodes`) must be able to handle "the model is
unavailable right now" without importing anything provider-specific — nodes
depend on interfaces, never on a concrete provider. Providers raise these; the
API layer turns them into a clean 503 with a real retry hint instead of a
500 with a stack trace.
"""

from typing import Optional


class LLMUnavailableError(RuntimeError):
    """The model could not answer this request (congestion, quota, outage)."""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMRateLimitedError(LLMUnavailableError):
    """Every candidate model is rate-limited / congested.

    Carries `retry_after_seconds` when the provider hinted a wait.
    """
