"""Agent1's long-term memory: the raw-fact Postgres table (docs §5.1).

Deliberately naive, because that is the baseline the demo exists to beat:

  * one flat table, ``agent1_facts``, exactly as the spec defines it;
  * no dedup, no update, no staleness logic — contradictions simply pile up as
    separate rows ("do not accidentally make Agent1 smarter than this");
  * retrieval = "pull every row for this session_id and dump it into context",
    unconditionally, with no relevance filtering.

When Postgres is unreachable the store falls back to an in-process list so the
demo still runs — but it says so (`backend` + `degraded_reason`), and the
dashboard shows which store actually served the turn. No silent pretending.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

_AGENT1_FACTS_DDL = """
CREATE TABLE IF NOT EXISTS agent1_facts (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    fact_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
)
"""

_BACKEND_POSTGRES = "postgres"
_BACKEND_MEMORY = "in-memory"


@dataclass
class FactRecord:
    id: int
    session_id: str
    fact_text: str
    created_at: datetime


class FactsStore(ABC):
    """Agent1's LTM contract. Both implementations behave identically on purpose:
    the *only* difference between them is resilience, never retrieval quality."""

    backend: str = _BACKEND_MEMORY
    degraded_reason: Optional[str] = None

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add_fact(self, session_id: str, fact_text: str) -> FactRecord: ...

    @abstractmethod
    async def list_facts(self, session_id: str) -> List[FactRecord]: ...

    async def aclose(self) -> None:  # pragma: no cover - nothing to release by default
        return None

    def stats(self) -> dict:
        return {"backend": self.backend, "degraded_reason": self.degraded_reason}


class InMemoryFactsStore(FactsStore):
    """Fallback used only when Postgres cannot be reached (and by the tests)."""

    backend = _BACKEND_MEMORY

    def __init__(self, degraded_reason: Optional[str] = None) -> None:
        self.degraded_reason = degraded_reason
        self._rows: List[FactRecord] = []
        self._next_id = 1

    async def initialize(self) -> None:
        return None

    async def add_fact(self, session_id: str, fact_text: str) -> FactRecord:
        record = FactRecord(
            id=self._next_id,
            session_id=session_id,
            fact_text=fact_text,
            created_at=datetime.now(timezone.utc),
        )
        self._next_id += 1
        self._rows.append(record)
        return record

    async def list_facts(self, session_id: str) -> List[FactRecord]:
        return [row for row in self._rows if row.session_id == session_id]


class PostgresFactsStore(FactsStore):
    """The real thing: asyncpg pool against POSTGRES_URL."""

    backend = _BACKEND_POSTGRES

    def __init__(self, dsn: str, connect_timeout: float = 5.0) -> None:
        self._dsn = dsn
        self._connect_timeout = connect_timeout
        self._pool = None

    async def initialize(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(dsn=self._dsn, timeout=self._connect_timeout, min_size=1, max_size=5)
        async with self._pool.acquire() as connection:
            await connection.execute(_AGENT1_FACTS_DDL)

    async def add_fact(self, session_id: str, fact_text: str) -> FactRecord:
        assert self._pool is not None, "initialize() must be awaited first"
        row = await self._pool.fetchrow(
            "INSERT INTO agent1_facts (session_id, fact_text) VALUES ($1, $2) "
            "RETURNING id, session_id, fact_text, created_at",
            session_id,
            fact_text,
        )
        return FactRecord(
            id=row["id"],
            session_id=row["session_id"],
            fact_text=row["fact_text"],
            created_at=row["created_at"],
        )

    async def list_facts(self, session_id: str) -> List[FactRecord]:
        assert self._pool is not None, "initialize() must be awaited first"
        rows = await self._pool.fetch(
            "SELECT id, session_id, fact_text, created_at FROM agent1_facts "
            "WHERE session_id = $1 ORDER BY id ASC",
            session_id,
        )
        return [
            FactRecord(id=r["id"], session_id=r["session_id"], fact_text=r["fact_text"], created_at=r["created_at"])
            for r in rows
        ]

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


async def open_facts_store(postgres_url: Optional[str]) -> FactsStore:
    """Use real Postgres when configured and reachable; otherwise fall back
    loudly (the reason lands in `stats()`/the dashboard, never in silence)."""
    if not postgres_url:
        return InMemoryFactsStore("POSTGRES_URL not configured")
    store = PostgresFactsStore(postgres_url)
    try:
        await store.initialize()
    except Exception as exc:
        return InMemoryFactsStore(f"postgres unreachable ({type(exc).__name__}: {exc})")
    store.degraded_reason = None
    return store
