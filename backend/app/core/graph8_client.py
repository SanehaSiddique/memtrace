"""graph8 MCP client — the one external tool layer both agents share (§4.3).

graph8 ships a real MCP server (586 tools as of this build). The demo
deliberately registers **three read tools only** (`g8_search_contacts`,
`g8_search_companies`, `g8_lookup_company`) so that the tool-selection decision
stays non-trivial while the raw result payloads stay big enough to matter —
which is exactly the point of §6.2 (Agent2 filters the raw payload; Agent1
pastes it into context verbatim).

Transports (both real graph8, no mocks):
  * ``mode="dev"``    — stdio: spawn the installed ``g8-mcp-server`` binary with
    ``G8_API_KEY`` / ``G8_API_URL`` in its environment.
  * ``mode="remote"`` — streamable HTTP against ``G8_MCP_URL`` with the API key
    as a Bearer header.

The client never raises on a tool failure: it returns a `ToolCallResult` with
``ok=False`` and the real error text, because a dead tool must not kill a turn
mid-demo — but the metrics collector records it, so a failed call is never
silently counted as an answer.

Lifecycle note: an MCP session's ``connect()`` and ``aclose()`` must be called
from the *same* asyncio task (anyio task-group affinity). FastAPI's lifespan
handler is one task, which is where the app does both; individual ``call_tool``
calls may come from any request task.
"""

import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from app.core.tracing import child_trace

# Domain-constrained tool registry (docs/IMPLEMENTATION_V2.md §3.2): curated
# from graph8's live MCP catalog (409 tools as of this build) into four
# CRM-shaped buckets. Both agents register the identical 50-tool set — Agent1
# injects all 50 schemas into every reasoning call unconditionally, Agent2's
# JEV routing (agent2/tool_routing.py) narrows it to at most one per turn.
# Picking between 3 tools proved nothing; picking correctly out of 50 does.
#
# g8_find_contacts/g8_find_companies were dropped from this registry: both
# have a *required* `filters` parameter typed as a bare array of untyped
# objects (`{"type": "array", "items": {"type": "object", "additionalProperties":
# true}}`) — no nested properties, no description, no example anywhere in the
# schema graph8 itself publishes. The model has no way to know what a filter
# object should contain, so calls to either tool reliably 422 ("Request
# validation failed"). Confirmed the trimming in `to_openai_schema()` isn't
# the cause — the raw, untrimmed schema is identical. Replaced with two
# tools that have fully concrete, unambiguous parameter schemas.
CONTACTS_TOOL_NAMES = [
    "g8_search_contacts",
    "g8_create_note",
    "g8_lookup_person",
    "g8_get_contact_detail",
    "g8_create_contact",
    "g8_update_contact",
    "g8_get_contact_activity",
    "g8_get_contact_deals",
    "g8_get_contact_company",
    "g8_list_hot_contacts",
    "g8_crm_get_list_contacts",
    "g8_get_contact_quotes",
    "g8_enrich_contacts",
    "g8_crm_list_contact_tasks",
    "g8_crm_assert_contact",
    "g8_crm_assert_contacts_batch",
    "g8_get_enrichment_job",
    "g8_list_enrichment_providers",
    "g8_crm_list_suppressions",
    "g8_add_to_list",
]

COMPANIES_DEALS_TOOL_NAMES = [
    "g8_search_companies",
    "g8_get_quote",
    "g8_lookup_company",
    "g8_crm_get_company",
    "g8_create_company",
    "g8_crm_update_company",
    "g8_get_company_contacts",
    "g8_get_company_deals",
    "g8_get_company_quotes",
    "g8_company_open_jobs",
    "g8_get_deals",
    "g8_get_deal",
    "g8_create_deal",
    "g8_update_deal",
    "g8_get_pipeline",
    "g8_delete_deal",
    "g8_crm_assert_company",
    "g8_crm_assert_companies_batch",
    "g8_crm_assert_deal",
    "g8_crm_list_company_columns",
]

SEQUENCES_CAMPAIGNS_TOOL_NAMES = [
    "g8_list_sequences",
    "g8_add_to_sequence",
    "g8_get_sequence_preview",
    "g8_get_sequence_analytics",
    "g8_sequence_get_sequence",
]

UTILITY_TOOL_NAMES = [
    "g8_get_activity_summary",
    "g8_get_activities",
    "g8_list_inbox",
    "g8_crm_list_duplicates",
    "g8_list_fields",
]

DEFAULT_TOOL_NAMES = (
    CONTACTS_TOOL_NAMES + COMPANIES_DEALS_TOOL_NAMES + SEQUENCES_CAMPAIGNS_TOOL_NAMES + UTILITY_TOOL_NAMES
)


@dataclass
class ToolCallResult:
    """Raw tool outcome, exactly as graph8 sent it."""

    name: str
    ok: bool
    raw_text: str = ""
    structured: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def raw_chars(self) -> int:
        return len(self.raw_text)


# graph8's real MCP schemas run 1.8-3.4k+ chars each (long descriptions, and
# per-field descriptions on every JSON Schema property) — fine for one tool,
# but 50 of them sent unconditionally (docs/IMPLEMENTATION_V2.md §3.2, Agent1's
# baseline) blows well past a real provider's per-request token cap. The LLM
# needs field names/types/required-ness to call a tool correctly, not a
# paragraph of prose per field, so the OpenAI-schema view trims verbosity;
# `description`/`input_schema` themselves stay untouched as the source of truth.
_TOOL_DESCRIPTION_MAX_CHARS = 60
_PARAM_DESCRIPTION_MAX_CHARS = 20


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _trim_param_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        trimmed = {}
        for key, value in schema.items():
            if key == "description" and isinstance(value, str):
                trimmed[key] = _trim_text(value, _PARAM_DESCRIPTION_MAX_CHARS)
            else:
                trimmed[key] = _trim_param_schema(value)
        return trimmed
    if isinstance(schema, list):
        return [_trim_param_schema(item) for item in schema]
    return schema


@dataclass
class Graph8ToolSpec:
    """One tool schema, kept provider-shaped and unedited (docs §5.2)."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _trim_text(self.description, _TOOL_DESCRIPTION_MAX_CHARS),
                "parameters": _trim_param_schema(self.input_schema) or {"type": "object", "properties": {}},
            },
        }


class Graph8MCPClient:
    """Thin MCP client: connect once, list schemas, call tools, stay honest."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = "dev",
        api_url: str = "https://be.graph8.com",
        server_url: str = "https://be.graph8.com/mcp/",
        command: str = "g8-mcp-server",
        tool_names: Optional[List[str]] = None,
        transport_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._mode = (mode or "dev").lower()
        self._api_url = api_url
        self._server_url = server_url
        self._command = command
        self._tool_names = list(tool_names or DEFAULT_TOOL_NAMES)
        self._transport_factory = transport_factory
        self._stack: Optional[AsyncExitStack] = None
        self._session: Any = None
        self._specs: List[Graph8ToolSpec] = []
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._call_count = 0
        self._error_count = 0

    # -- lifecycle --------------------------------------------------------------

    @asynccontextmanager
    async def _stdio_session(self) -> AsyncIterator[Any]:
        import os

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["G8_API_KEY"] = self._api_key or env.get("G8_API_KEY", "")
        env["G8_API_URL"] = self._api_url
        params = StdioServerParameters(command=self._command, args=[], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session

    @asynccontextmanager
    async def _http_session(self) -> AsyncIterator[Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        async with streamablehttp_client(self._server_url, headers=headers) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                yield session

    def _session_context(self) -> Any:
        if self._transport_factory is not None:
            return self._transport_factory()
        return self._stdio_session() if self._mode == "dev" else self._http_session()

    async def connect(self) -> bool:
        """Connect and load the registered tool schemas. Idempotent; never raises."""
        if self._session is not None:
            return True
        try:
            self._stack = AsyncExitStack()
            self._session = await self._stack.enter_async_context(self._session_context())
            await self._session.initialize()
            tools = (await self._session.list_tools()).tools
            specs = {t.name: t for t in tools}
            missing = [name for name in self._tool_names if name not in specs]
            self._specs = [
                Graph8ToolSpec(
                    name=specs[name].name,
                    description=getattr(specs[name], "description", "") or "",
                    input_schema=getattr(specs[name], "inputSchema", None) or {},
                )
                for name in self._tool_names
                if name in specs
            ]
            self.available = bool(self._specs)
            if missing:
                self.unavailable_reason = f"tools not exposed by server: {missing}"
            return self.available
        except Exception as exc:
            self.available = False
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            await self.aclose()
            return False

    async def aclose(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        self.available = False
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                pass

    # -- tools ------------------------------------------------------------------

    @property
    def specs(self) -> List[Graph8ToolSpec]:
        return list(self._specs)

    def tool_schemas(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """OpenAI-style schemas for the requested tools (all registered ones by
        default). Agent1 sends all of these on every request; Agent2 sends at
        most one — the token difference is the headline §6.3 metric."""
        wanted = set(names) if names is not None else None
        return [spec.to_openai_schema() for spec in self._specs if wanted is None or spec.name in wanted]

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        agent_id: str = "",
        run_group_id: str = "",
    ) -> ToolCallResult:
        """Call one registered tool and return its **raw** payload (§4.3)."""
        started = time.perf_counter()
        self._call_count += 1
        with child_trace(
            name=f"tool.{name}",
            run_type="tool",
            agent_id=agent_id,
            run_group_id=run_group_id,
            component="tool",
            call_type="tool_call",
            metadata={"tool": name, "arguments": arguments or {}, "transport": self._mode},
        ) as run:
            if self._session is None:
                self._error_count += 1
                result = ToolCallResult(
                    name=name,
                    ok=False,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    error=f"graph8 MCP not connected: {self.unavailable_reason or 'no session'}",
                )
                self._finish_trace(run, result)
                return result
            try:
                response = await self._session.call_tool(name, arguments or {})
            except Exception as exc:
                self._error_count += 1
                result = ToolCallResult(
                    name=name,
                    ok=False,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._finish_trace(run, result)
                return result

            raw_text = "".join(
                getattr(block, "text", "") or "" for block in (getattr(response, "content", None) or [])
            )
            structured = self._parse_payload(raw_text)
            is_error = bool(getattr(response, "isError", False))
            if is_error:
                self._error_count += 1
            result = ToolCallResult(
                name=name,
                ok=not is_error,
                raw_text=raw_text,
                structured=structured,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                error=None if not is_error else (raw_text[:400] or "tool returned an error"),
            )
            self._finish_trace(run, result)
            return result

    @staticmethod
    def _parse_payload(raw_text: str) -> Optional[Dict[str, Any]]:
        import json

        try:
            parsed = json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else {"items": parsed}

    @staticmethod
    def _finish_trace(run: Any, result: ToolCallResult) -> None:
        try:
            run.outputs = {
                "ok": result.ok,
                "raw_chars": result.raw_chars,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        except Exception:
            pass  # tracing must never break a real tool call

    def stats(self) -> Dict[str, Any]:
        return {
            "transport": self._mode,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "registered_tools": [spec.name for spec in self._specs] or list(self._tool_names),
            "tool_calls": self._call_count,
            "tool_errors": self._error_count,
        }

