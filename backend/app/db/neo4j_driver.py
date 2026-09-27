"""Agent2's long-term memory: the Neo4j fact graph (docs §6.1).

Shape, exactly as the spec defines it:

    (:Session {id})-[:HAS_FACT]->(:Fact {id, text, status, created_at})
    (:Fact)-[:ABOUT]->(:Entity {name, session_id})
    (:Fact {status:"stale"})-[:SUPERSEDED_BY]->(:Fact {status:"active"})

Stale facts are **never deleted** — they are marked and linked, so the graph
keeps the history that explains why the current answer is current. That is the
Java→Python (MongoDB→PostgreSQL in this demo's story) behavior the dashboard
renders.

As with Postgres, an in-process implementation of the same contract exists for
when Neo4j is not provisioned (and for hermetic tests); the store reports which
backend it is via `backend`/`stats()`.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STATUS_ACTIVE = "active"
STATUS_STALE = "stale"

_BACKEND_NEO4J = "neo4j"
_BACKEND_MEMORY = "in-memory"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return f"fact_{uuid.uuid4().hex[:12]}"


@dataclass
class FactNode:
    """One fact in the graph, with the entity it is about and its status."""

    id: str
    session_id: str
    text: str
    entity: str
    status: str = STATUS_ACTIVE
    created_at: str = field(default_factory=_iso_now)
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None


class FactGraphStore(ABC):
    backend: str = _BACKEND_MEMORY
    degraded_reason: Optional[str] = None

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add_fact(self, session_id: str, text: str, entity: str) -> FactNode: ...

    @abstractmethod
    async def find_active_facts(self, session_id: str, entity: str) -> List[FactNode]: ...

    @abstractmethod
    async def mark_stale(self, old_fact_id: str, new_fact_id: str, session_id: str) -> None: ...

    @abstractmethod
    async def session_facts(self, session_id: str) -> List[FactNode]: ...

    @abstractmethod
    async def graph_snapshot(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]: ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None

    def stats(self) -> dict:
        return {"backend": self.backend, "degraded_reason": self.degraded_reason}


class InMemoryFactGraph(FactGraphStore):
    """Same contract as the Neo4j store, in process memory (tests / no DB)."""

    backend = _BACKEND_MEMORY

    def __init__(self, degraded_reason: Optional[str] = None) -> None:
        self.degraded_reason = degraded_reason
        self._facts: Dict[str, FactNode] = {}

    async def initialize(self) -> None:
        return None

    async def add_fact(self, session_id: str, text: str, entity: str) -> FactNode:
        node = FactNode(id=_new_id(), session_id=session_id, text=text, entity=entity)
        self._facts[node.id] = node
        return node

    async def find_active_facts(self, session_id: str, entity: str) -> List[FactNode]:
        return [
            fact
            for fact in self._facts.values()
            if fact.session_id == session_id and fact.entity == entity and fact.status == STATUS_ACTIVE
        ]

    async def mark_stale(self, old_fact_id: str, new_fact_id: str, session_id: str) -> None:
        old = self._facts.get(old_fact_id)
        new = self._facts.get(new_fact_id)
        if old is None or new is None or old.session_id != session_id:
            return
        old.status = STATUS_STALE
        old.superseded_by = new_fact_id
        new.supersedes = old_fact_id

    async def session_facts(self, session_id: str) -> List[FactNode]:
        facts = [f for f in self._facts.values() if f.session_id == session_id]
        facts.sort(key=lambda f: f.created_at)
        return facts

    async def graph_snapshot(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        facts = await self.session_facts(session_id)
        entities = sorted({f.entity for f in facts})
        nodes: List[Dict[str, Any]] = [
            {"id": f.id, "label": "Fact", "text": f.text, "status": f.status, "entity": f.entity, "created_at": f.created_at}
            for f in facts
        ] + [{"id": f"entity::{name}", "label": "Entity", "name": name} for name in entities]
        edges: List[Dict[str, Any]] = []
        for fact in facts:
            edges.append({"source": fact.id, "target": f"entity::{fact.entity}", "type": "ABOUT"})
            if fact.superseded_by:
                edges.append({"source": fact.id, "target": fact.superseded_by, "type": "SUPERSEDED_BY"})
        return {"nodes": nodes, "edges": edges}


class Neo4jFactGraph(FactGraphStore):
    """The real graph store: the official async Neo4j driver + Cypher."""

    backend = _BACKEND_NEO4J

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._auth = (user, password)
        self._driver: Any = None

    async def initialize(self) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)
        await self._driver.verify_connectivity()
        await self._run("CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE", {})

    async def _run(self, cypher: str, parameters: Dict[str, Any]) -> List[Any]:
        assert self._driver is not None, "initialize() must be awaited first"
        records, _summary, _keys = await self._driver.execute_query(cypher, parameters)
        return list(records)

    async def add_fact(self, session_id: str, text: str, entity: str) -> FactNode:
        node = FactNode(id=_new_id(), session_id=session_id, text=text, entity=entity)
        await self._run(
            """
            MERGE (s:Session {id: $session_id})
            MERGE (e:Entity {session_id: $session_id, name: $entity})
            CREATE (f:Fact {id: $id, session_id: $session_id, text: $text, status: $status, created_at: $created_at})
            CREATE (s)-[:HAS_FACT]->(f)
            CREATE (f)-[:ABOUT]->(e)
            """,
            {
                "session_id": session_id,
                "entity": entity,
                "id": node.id,
                "text": node.text,
                "status": node.status,
                "created_at": node.created_at,
            },
        )
        return node

    async def find_active_facts(self, session_id: str, entity: str) -> List[FactNode]:
        records = await self._run(
            """
            MATCH (f:Fact {session_id: $session_id, status: $status})-[:ABOUT]->(e:Entity {name: $entity})
            RETURN f.id AS id, f.text AS text, f.created_at AS created_at
            ORDER BY f.created_at ASC
            """,
            {"session_id": session_id, "entity": entity, "status": STATUS_ACTIVE},
        )
        return [
            FactNode(id=r["id"], session_id=session_id, text=r["text"], entity=entity, created_at=r["created_at"] or "")
            for r in records
        ]

    async def mark_stale(self, old_fact_id: str, new_fact_id: str, session_id: str) -> None:
        """Never destroy a stale fact — mark it and link it (§6.1, §6.4)."""
        await self._run(
            """
            MATCH (old:Fact {id: $old_id, session_id: $session_id})
            MATCH (new:Fact {id: $new_id, session_id: $session_id})
            SET old.status = $stale
            CREATE (old)-[:SUPERSEDED_BY]->(new)
            """,
            {"old_id": old_fact_id, "new_id": new_fact_id, "session_id": session_id, "stale": STATUS_STALE},
        )

    async def session_facts(self, session_id: str) -> List[FactNode]:
        records = await self._run(
            """
            MATCH (f:Fact {session_id: $session_id})
            OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
            OPTIONAL MATCH (f)-[:SUPERSEDED_BY]->(n:Fact)
            OPTIONAL MATCH (p:Fact)-[:SUPERSEDED_BY]->(f)
            RETURN f.id AS id, f.text AS text, f.status AS status, f.created_at AS created_at,
                   coalesce(e.name, '') AS entity, n.id AS superseded_by, p.id AS supersedes
            ORDER BY f.created_at ASC
            """,
            {"session_id": session_id},
        )
        return [
            FactNode(
                id=r["id"],
                session_id=session_id,
                text=r["text"],
                entity=r["entity"] or "",
                status=r["status"] or STATUS_ACTIVE,
                created_at=r["created_at"] or "",
                superseded_by=r["superseded_by"],
                supersedes=r["supersedes"],
            )
            for r in records
        ]

    async def graph_snapshot(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        facts = await self.session_facts(session_id)
        entities = sorted({f.entity for f in facts if f.entity})
        nodes: List[Dict[str, Any]] = [
            {
                "id": f.id,
                "label": "Fact",
                "text": f.text,
                "status": f.status,
                "entity": f.entity,
                "created_at": f.created_at,
                "superseded_by": f.superseded_by,
            }
            for f in facts
        ] + [{"id": f"entity::{name}", "label": "Entity", "name": name} for name in entities]
        edges: List[Dict[str, Any]] = []
        for fact in facts:
            if fact.entity:
                edges.append({"source": fact.id, "target": f"entity::{fact.entity}", "type": "ABOUT"})
            if fact.superseded_by:
                edges.append({"source": fact.id, "target": fact.superseded_by, "type": "SUPERSEDED_BY"})
        return {"nodes": nodes, "edges": edges}

    async def aclose(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None


async def open_fact_graph(uri: Optional[str], user: str, password: str) -> FactGraphStore:
    """Real Neo4j when configured and reachable; otherwise fall back loudly."""
    if not uri:
        return InMemoryFactGraph("NEO4J_URI not configured")
    store = Neo4jFactGraph(uri, user, password)
    try:
        await store.initialize()
    except Exception as exc:
        await store.aclose()
        return InMemoryFactGraph(f"neo4j unreachable ({type(exc).__name__}: {exc})")
    store.degraded_reason = None
    return store


