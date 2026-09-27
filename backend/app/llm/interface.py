"""LLM client interface. Callers depend on this, never on a concrete provider."""

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional, Tuple

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One function/tool call the model asked for, as the provider sent it."""

    id: str = ""
    name: str
    arguments: str = ""  # raw JSON string exactly as returned by the provider

    def parsed_arguments(self) -> dict:
        """Tolerant JSON parse — providers occasionally return an empty string
        (or a JSON-encoded string) for a no-argument call."""
        import json

        if not self.arguments:
            return {}
        try:
            parsed = json.loads(self.arguments)
        except (ValueError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                again = json.loads(parsed)
                return again if isinstance(again, dict) else {}
            except (ValueError, TypeError):
                return {}
        return {}


class ChatCompletion(BaseModel):
    """Provider-neutral result of a (possibly tool-calling) chat completion."""

    content: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    model: str = ""


class BaseLLMClient(ABC):
    """OpenAI-compatible chat + embeddings client."""

    is_live: bool = False

    # True when `embed()` is computed locally (no provider request, no quota
    # consumed). OpenRouter has no free embedding model, so its client hashes
    # text into a vector locally instead of billing an embeddings endpoint.
    embeddings_are_local: bool = False

    # Provider id and model id, used for per-model usage/cost accounting.
    provider: str = "unknown"
    model_name: str = "unknown"

    @abstractmethod
    async def chat(self, system: str, user: str, temperature: float = 0.0) -> str: ...

    @abstractmethod
    async def embed(self, text: str) -> List[float]: ...

    async def chat_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Tuple[str, dict, str]:
        """Like `chat()`, but also returns the provider-reported token usage and
        the model that actually answered (which may differ from the configured
        one when a provider falls back), so token accounting stays honest.

        Providers that report usage override this; the default reports none.
        """
        return await self.chat(system, user, temperature), {}, self.model_name

    async def chat_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> "ChatCompletion":
        """Full OpenAI-style chat completion with optional tool schemas.

        This is the entry point the agent-comparison demo needs (it must count
        exactly how many tool schemas were in the prompt, and read back the
        model's raw `tool_calls`). Providers that cannot honor it raise
        `LLMUnavailableError`; callers treat that like any other provider
        outage rather than crashing the turn.
        """
        from app.llm.errors import LLMUnavailableError

        raise LLMUnavailableError(f"{type(self).__name__} does not support tool-calling chat completions")

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(t) for t in texts]

    async def chat_completion_stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> "ChatCompletion":
        """Like `chat_completion`, but calls `on_token` for each content delta
        as it streams in (docs/IMPLEMENTATION_V2.md §5.2 `llm_final_answer_token`).

        Providers that can stream override this with a real SSE implementation.
        This default degrades gracefully for any that can't: one non-streaming
        call, then `on_token` fires once with the full content — callers still
        get a real answer, just without incremental visibility.
        """
        completion = await self.chat_completion(messages, tools=tools, temperature=temperature, model=model)
        if on_token and completion.content:
            await on_token(completion.content)
        return completion
