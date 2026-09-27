"""Parsing for the OpenAI-compatible chat-completions wire format.

Shared by every provider client that speaks that format (OpenAI-compatible,
OpenRouter, Groq) so tool-call extraction and usage normalization exist once.
Groq and OpenRouter return the same `choices[0].message` shape; the only
difference worth normalizing is that some providers return `content: null` for
a pure tool-call response.
"""

from typing import Any, Dict

from app.llm.interface import ChatCompletion, ToolCall


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
