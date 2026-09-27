"""Persistent, server-side client for Graph8's remote MCP endpoint.

The remote endpoint authenticates headless applications with an API key in a
Bearer header. It advertises a compact initial tool set and can activate more
tools in the same session through ``g8_tool_search``.
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, types
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client


class Graph8ConfigurationError(RuntimeError):
    pass


class Graph8ToolPermissionError(PermissionError):
    pass


class Graph8MCPClient:
    def __init__(self, api_key: Optional[str], url: str, allow_mutations: bool = False) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.url = url
        self.allow_mutations = allow_mutations
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._connect_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def _connect(self) -> ClientSession:
        if not self.configured:
            raise Graph8ConfigurationError("Graph8 MCP is not configured. Set G8_API_KEY in memtrace/.env.")
        if self._session is not None:
            return self._session

        async with self._connect_lock:
            if self._session is not None:
                return self._session

            stack = AsyncExitStack()
            try:
                http_client = create_mcp_http_client(
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                )
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(self.url, http_client=http_client)
                )
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
            except Exception:
                await stack.aclose()
                raise

            self._stack = stack
            self._session = session
            return session

    async def close(self) -> None:
        # Streamable HTTP owns an AnyIO task group; shut it down from the same
        # app lifespan that opened it.
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def _all_tools(self) -> list[types.Tool]:
        session = await self._connect()
        tools: list[types.Tool] = []
        cursor: Optional[str] = None
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            page = await session.list_tools(params=params)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                return tools

    @staticmethod
    def _serialize_tool(tool: types.Tool) -> dict[str, Any]:
        payload = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        annotations = tool.annotations
        payload["memtrace_access"] = (
            "read_only" if annotations is not None and annotations.read_only_hint is True else "mutation_guarded"
        )
        return payload

    async def list_tools(self) -> list[dict[str, Any]]:
        async with self._call_lock:
            return [self._serialize_tool(tool) for tool in await self._all_tools()]

    async def status(self) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "connected": False,
                "url": self.url,
                "mutations_enabled": self.allow_mutations,
                "tool_count": 0,
            }
        tools = await self.list_tools()
        return {
            "configured": True,
            "connected": True,
            "url": self.url,
            "mutations_enabled": self.allow_mutations,
            "tool_count": len(tools),
        }

    async def discover(self, query: Optional[str]) -> Optional[dict[str, Any]]:
        if not query or not query.strip():
            return None
        # This call must share the persistent session with the following
        # tools/list request; activated tools are session-scoped.
        return await self.call_tool("g8_tool_search", {"query": query.strip()})

    async def call_tool(self, name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        async with self._call_lock:
            session = await self._connect()
            tools = {tool.name: tool for tool in await self._all_tools()}
            tool = tools.get(name)
            if tool is None:
                raise KeyError(
                    f"Graph8 tool '{name}' is not currently advertised. "
                    "Discover it first with GET /integrations/graph8/tools?query=..."
                )

            is_read_only = tool.annotations is not None and tool.annotations.read_only_hint is True
            if not is_read_only and not self.allow_mutations:
                raise Graph8ToolPermissionError(
                    f"Graph8 tool '{name}' is not marked read-only. "
                    "Set G8_MCP_ALLOW_MUTATIONS=true only after securing the MEMTRACE API."
                )

            result = await session.call_tool(name, arguments or {})
            return result.model_dump(mode="json", by_alias=True, exclude_none=True)
