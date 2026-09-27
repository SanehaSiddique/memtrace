"""Parsing for the OpenAI-compatible chat-completions wire format.

Shared by every provider client that speaks that format (OpenAI-compatible,
OpenRouter, Groq) so tool-call extraction and usage normalization exist once.
Groq and OpenRouter return the same `choices[0].message` shape; the only
difference worth normalizing is that some providers return `content: null` for
a pure tool-call response.
"""

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from app.llm.interface import ChatCompletion, ToolCall

OnToken = Callable[[str], Awaitable[None]]


async def _default_on_token(_token: str) -> None:
    return None


class StreamRateLimited(Exception):
    """A streaming attempt hit HTTP 429 before any content arrived — the
    caller decides how to turn this into its own rate-limit error type."""

    def __init__(self, detail: str, retry_after_seconds: Optional[float] = None) -> None:
        super().__init__(detail)
        self.retry_after_seconds = retry_after_seconds


def seconds_from_retry_after(value: Optional[str]) -> Optional[float]:
    """`Retry-After` is either delta-seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        import time
        from email.utils import parsedate_to_datetime

        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except Exception:
        return None


async def stream_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    fallback_model: str,
    on_token: Optional[OnToken] = None,
) -> ChatCompletion:
    """Stream an OpenAI-compatible SSE chat completion (docs/IMPLEMENTATION_V2.md
    §5.2 `llm_final_answer_token`), calling `on_token` for each content delta as
    it arrives, and returning the fully assembled `ChatCompletion` once the
    stream ends — same shape `parse_chat_completion` produces for a
    non-streaming call, so callers account for it identically either way.
    Shared by every provider that speaks this wire format (OpenAI-compatible,
    OpenRouter, Groq).
    """
    from app.llm.errors import LLMUnavailableError

    on_token = on_token or _default_on_token
    stream_payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}

    content_parts: List[str] = []
    tool_calls_acc: Dict[int, Dict[str, str]] = {}
    usage: Dict[str, Any] = {}
    model = fallback_model

    async with client.stream("POST", url, headers=headers, json=stream_payload) as response:
        if response.status_code == 429:
            body = (await response.aread()).decode(errors="replace")
            raise StreamRateLimited(
                f"rate-limited (HTTP 429): {body[:300]}",
                seconds_from_retry_after(response.headers.get("Retry-After")),
            )
        if response.status_code >= 400:
            body = (await response.aread()).decode(errors="replace")
            raise LLMUnavailableError(f"HTTP {response.status_code}: {body[:300]}")

        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except ValueError:
                continue

            if chunk.get("model"):
                model = chunk["model"]
            if chunk.get("usage"):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}

            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
                await on_token(piece)

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["arguments"] += fn["arguments"]

    tool_calls = [
        ToolCall(id=acc["id"], name=acc["name"], arguments=acc["arguments"])
        for acc in tool_calls_acc.values()
        if acc["name"]
    ]
    return ChatCompletion(content="".join(content_parts), tool_calls=tool_calls, usage=usage, model=model)


def parse_chat_completion(data: Dict[str, Any], fallback_model: str = "") -> ChatCompletion:
    choices = data.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    tool_calls = [
        ToolCall(
            id=str(call.get("id") or ""),
            name=str((call.get("function") or {}).get("name") or ""),
            arguments=str((call.get("function") or {}).get("arguments") or ""),
        )
        for call in (message.get("tool_calls") or [])
        if (call.get("function") or {}).get("name")
    ]
    return ChatCompletion(
        content=(message.get("content") or ""),
        tool_calls=tool_calls,
        usage=data.get("usage") or {},
        model=str(data.get("model") or fallback_model),
    )
