"""Durable chat + turn-metrics history for the comparison view.

The UI's message list lived only in React state, so a page reload wiped the
transcript even though the graph and memory survived. This persists each turn
so the benchmark can be reloaded, shared, or audited after the fact.

Design notes
------------
* One row per (session, agent, role). The user's prompt is stored once *per
  agent* rather than shared, because each agent panel renders its own complete
  transcript — sharing it would force the client to interleave two agents'
  histories and get the ordering wrong.
* `seq` is an autoincrement so ordering is total and stable even when two rows
  land in the same millisecond (which they do: both agents answer a turn
  concurrently). Ordering by `turn_index, role` alone would tie.
* `metrics` holds the full `TurnMetrics` dump, so rehydrating the dashboard
  needs no second store — the same row that renders a chat bubble also restores
  the per-turn cost/token figures.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite


class ChatHistoryRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    run_group_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    metrics TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_turns(session_id, agent_id, seq);")
            await db.commit()

    async def record(
        self,
        session_id: str,
        agent_id: str,
        run_group_id: str,
        turn_index: int,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        row_id = f"chat_{uuid.uuid4().hex[:12]}"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO chat_turns
                    (id, session_id, agent_id, run_group_id, turn_index, role, content, tool_calls, metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    session_id,
                    agent_id,
                    run_group_id,
                    int(turn_index),
                    role,
                    content,
                    json.dumps(tool_calls or []),
                    json.dumps(metrics) if metrics is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
        return row_id

    async def list_session(
        self, session_id: str, agent_id: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """History for one session, optionally narrowed to a single agent."""
        query = (
            "SELECT id, session_id, agent_id, run_group_id, turn_index, role, content, "
            "tool_calls, metrics, created_at FROM chat_turns WHERE session_id = ?"
        )
        params: List[Any] = [session_id]
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY seq ASC LIMIT ?"
        params.append(int(limit))

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        messages: List[Dict[str, Any]] = []
        for r in rows:
            # A malformed JSON blob must not take down the whole transcript —
            # one bad row should cost that row, not the page load.
            try:
                tool_calls = json.loads(r[7]) or []
            except (TypeError, ValueError):
                tool_calls = []
            try:
                metrics = json.loads(r[8]) if r[8] else None
            except (TypeError, ValueError):
                metrics = None
            messages.append(
                {
                    "id": r[0],
                    "session_id": r[1],
                    "agent_id": r[2],
                    "run_group_id": r[3],
                    "turn_index": r[4],
                    "role": r[5],
                    "content": r[6],
                    "tool_calls": tool_calls,
                    "metrics": metrics,
                    "created_at": r[9],
                }
            )
        return messages

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Every session with history, most recently active first — powers the
        session switcher so an earlier benchmark can be reopened."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT session_id, MAX(created_at) AS last_at, COUNT(DISTINCT run_group_id) AS turns
                FROM chat_turns GROUP BY session_id ORDER BY last_at DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"session_id": r[0], "last_activity": r[1], "turns": r[2]} for r in rows]

    async def clear(self, session_id: Optional[str] = None) -> int:
        """Delete history for one session, or all of it. Returns rows removed."""
        async with aiosqlite.connect(self.db_path) as db:
            if session_id:
                cur = await db.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
            else:
                cur = await db.execute("DELETE FROM chat_turns")
            removed = cur.rowcount or 0
            await db.commit()
        return removed
