"""Memory and graph repository interface plus the test-only SQLite adapter.

This is the single seam between the rest of MEMTRACE (LangGraph nodes, services,
retrieval, context building) and the underlying storage engine. Everything above
this layer depends only on `BaseMemoryRepository`. Production uses Neo4j;
SQLite remains a lightweight hermetic adapter for unit tests.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from app.memory.models import (
    Memory,
    MemoryEvent,
    MemoryRelationship,
    MemoryStatus,
    RelationType,
)

_MEMORY_COLUMNS = """
    id, agent_id, memory_type, subject, predicate, object, content,
    confidence, status, created_at, valid_from, valid_until,
    supersedes_memory_id, superseded_by_memory_id, source_event_id,
    provenance_json, metadata_json, embedding_json
"""


class BaseMemoryRepository(ABC):
    """Abstract interface for storing memories, graph edges, and events."""

    @abstractmethod
    async def initialize(self) -> None: ...

    async def close(self) -> None:
        """Release repository resources. Stateless adapters need no cleanup."""

    @abstractmethod
    async def save_event(self, event: MemoryEvent) -> MemoryEvent: ...

    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[MemoryEvent]: ...

    @abstractmethod
    async def create_memory(self, memory: Memory, embedding: Optional[List[float]] = None) -> Memory: ...

    @abstractmethod
    async def update_memory(self, memory: Memory) -> Memory: ...

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Memory]: ...

    @abstractmethod
    async def list_memories(
        self,
        agent_id: str,
        status: Optional[MemoryStatus] = None,
        subject: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]: ...

    @abstractmethod
    async def get_active_memories(self, agent_id: str, subject: Optional[str] = None) -> List[Memory]: ...

    @abstractmethod
    async def get_historical_memories(self, agent_id: str, subject: Optional[str] = None) -> List[Memory]: ...

    @abstractmethod
    async def archive_memory(self, memory_id: str, reason: str = "") -> Optional[Memory]: ...

    @abstractmethod
    async def delete_memory(self, memory_id: str, reason: str = "") -> Optional[Memory]: ...

    @abstractmethod
    async def add_relationship(self, relationship: MemoryRelationship) -> MemoryRelationship: ...

    @abstractmethod
    async def get_relationships_for_node(
        self,
        node_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[MemoryRelationship]: ...

    @abstractmethod
    async def get_related_memories(
        self,
        node_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Tuple[MemoryRelationship, Optional[Memory]]]: ...

    @abstractmethod
    async def get_memory_history(self, memory_id: str) -> List[Memory]: ...

    @abstractmethod
    async def get_provenance(self, memory_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def find_existing_memories_by_triple(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        status: Optional[MemoryStatus] = None,
    ) -> List[Memory]: ...

    @abstractmethod
    async def search_semantic(
        self,
        agent_id: str,
        query_embedding: List[float],
        limit: int = 10,
        status: Optional[MemoryStatus] = None,
    ) -> List[Tuple[Memory, float]]: ...


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteMemoryRepository(BaseMemoryRepository):
    """SQLite implementation storing memories, events, and graph relationships."""

    def __init__(self, db_path: str = "memtrace.db"):
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA foreign_keys=ON;")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    valid_from TEXT,
                    valid_until TEXT,
                    supersedes_memory_id TEXT,
                    superseded_by_memory_id TEXT,
                    source_event_id TEXT,
                    provenance_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT
                );
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_agent_status
                ON memories(agent_id, status);
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_subject_pred
                ON memories(agent_id, subject, predicate);
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_source
                ON relationships(source_id);
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_target
                ON relationships(target_id);
            """)
            await db.commit()

    # -- events -----------------------------------------------------------

    async def save_event(self, event: MemoryEvent) -> MemoryEvent:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, conversation_id, agent_id, timestamp, speaker, content, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.conversation_id,
                    event.agent_id,
                    event.timestamp.isoformat(),
                    event.speaker,
                    event.content,
                    json.dumps(event.metadata),
                ),
            )
            await db.commit()
        return event

    async def get_event(self, event_id: str) -> Optional[MemoryEvent]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT event_id, conversation_id, agent_id, timestamp, speaker, content, metadata_json
                FROM events WHERE event_id = ?
                """,
                (event_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return None
        return MemoryEvent(
            event_id=row[0],
            conversation_id=row[1],
            agent_id=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            speaker=row[4],
            content=row[5],
            metadata=json.loads(row[6]),
        )

    # -- memories -----------------------------------------------------------

    def _row_to_memory(self, row: tuple) -> Memory:
        return Memory(
            id=row[0],
            agent_id=row[1],
            memory_type=row[2],
            subject=row[3],
            predicate=row[4],
            object=row[5],
            content=row[6],
            confidence=row[7],
            status=MemoryStatus(row[8]),
            created_at=datetime.fromisoformat(row[9]),
            valid_from=datetime.fromisoformat(row[10]) if row[10] else None,
            valid_until=datetime.fromisoformat(row[11]) if row[11] else None,
            supersedes_memory_id=row[12],
            superseded_by_memory_id=row[13],
            source_event_id=row[14],
            provenance=json.loads(row[15]),
            metadata=json.loads(row[16]),
        )

    async def create_memory(self, memory: Memory, embedding: Optional[List[float]] = None) -> Memory:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"""
                INSERT INTO memories ({_MEMORY_COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.agent_id,
                    memory.memory_type.value if hasattr(memory.memory_type, "value") else memory.memory_type,
                    memory.subject,
                    memory.predicate,
                    memory.object,
                    memory.content,
                    memory.confidence,
                    memory.status.value if hasattr(memory.status, "value") else memory.status,
                    memory.created_at.isoformat(),
                    memory.valid_from.isoformat() if memory.valid_from else None,
                    memory.valid_until.isoformat() if memory.valid_until else None,
                    memory.supersedes_memory_id,
                    memory.superseded_by_memory_id,
                    memory.source_event_id,
                    json.dumps(memory.provenance),
                    json.dumps(memory.metadata),
                    json.dumps(embedding) if embedding else None,
                ),
            )
            await db.commit()
        return memory

    async def update_memory(self, memory: Memory) -> Memory:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE memories SET
                    agent_id = ?, memory_type = ?, subject = ?, predicate = ?, object = ?,
                    content = ?, confidence = ?, status = ?, created_at = ?, valid_from = ?,
                    valid_until = ?, supersedes_memory_id = ?, superseded_by_memory_id = ?,
                    source_event_id = ?, provenance_json = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    memory.agent_id,
                    memory.memory_type.value if hasattr(memory.memory_type, "value") else memory.memory_type,
                    memory.subject,
                    memory.predicate,
                    memory.object,
                    memory.content,
                    memory.confidence,
                    memory.status.value if hasattr(memory.status, "value") else memory.status,
                    memory.created_at.isoformat(),
                    memory.valid_from.isoformat() if memory.valid_from else None,
                    memory.valid_until.isoformat() if memory.valid_until else None,
                    memory.supersedes_memory_id,
                    memory.superseded_by_memory_id,
                    memory.source_event_id,
                    json.dumps(memory.provenance),
                    json.dumps(memory.metadata),
                    memory.id,
                ),
            )
            await db.commit()
        return memory

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?",
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    async def list_memories(
        self,
        agent_id: str,
        status: Optional[MemoryStatus] = None,
        subject: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        query = f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE agent_id = ?"
        params: List[Any] = [agent_id]

        if status:
            query += " AND status = ?"
            params.append(status.value if hasattr(status, "value") else status)
        if subject:
            query += " AND LOWER(subject) = LOWER(?)"
            params.append(subject)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_memory(r) for r in rows]

    async def get_active_memories(self, agent_id: str, subject: Optional[str] = None) -> List[Memory]:
        return await self.list_memories(agent_id, status=MemoryStatus.ACTIVE, subject=subject, limit=1000)

    async def get_historical_memories(self, agent_id: str, subject: Optional[str] = None) -> List[Memory]:
        return await self.list_memories(agent_id, status=MemoryStatus.HISTORICAL, subject=subject, limit=1000)

    async def archive_memory(self, memory_id: str, reason: str = "") -> Optional[Memory]:
        memory = await self.get_memory(memory_id)
        if not memory:
            return None
        memory.status = MemoryStatus.ARCHIVED
        memory.valid_until = memory.valid_until or datetime.now(timezone.utc)
        if reason:
            memory.metadata = {**memory.metadata, "archive_reason": reason}
        return await self.update_memory(memory)

    async def delete_memory(self, memory_id: str, reason: str = "") -> Optional[Memory]:
        """Soft-delete: keep the row (for provenance/history) but mark status DELETED."""
        memory = await self.get_memory(memory_id)
        if not memory:
            return None
        memory.status = MemoryStatus.DELETED
        if reason:
            memory.metadata = {**memory.metadata, "delete_reason": reason}
        return await self.update_memory(memory)

    async def find_existing_memories_by_triple(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        status: Optional[MemoryStatus] = None,
    ) -> List[Memory]:
        query = f"""
            SELECT {_MEMORY_COLUMNS}
            FROM memories
            WHERE agent_id = ?
              AND LOWER(subject) = LOWER(?)
              AND LOWER(predicate) = LOWER(?)
        """
        params: List[Any] = [agent_id, subject, predicate]
        if status:
            query += " AND status = ?"
            params.append(status.value if hasattr(status, "value") else status)

        query += " ORDER BY created_at DESC"
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_memory(r) for r in rows]

    async def search_semantic(
        self,
        agent_id: str,
        query_embedding: List[float],
        limit: int = 10,
        status: Optional[MemoryStatus] = None,
    ) -> List[Tuple[Memory, float]]:
        """Brute-force cosine similarity over stored embeddings.

        Fine at MVP/demo scale (dozens-hundreds of memories); would move to a
        vector index (pgvector/FAISS) if the memory store grew large.
        """
        query = f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE agent_id = ? AND embedding_json IS NOT NULL"
        params: List[Any] = [agent_id]
        if status:
            query += " AND status = ?"
            params.append(status.value if hasattr(status, "value") else status)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()

        scored: List[Tuple[Memory, float]] = []
        for row in rows:
            embedding = json.loads(row[17]) if row[17] else None
            if not embedding:
                continue
            mem = self._row_to_memory(row)
            scored.append((mem, _cosine_similarity(query_embedding, embedding)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    # -- graph / relationships -----------------------------------------------

    async def add_relationship(self, relationship: MemoryRelationship) -> MemoryRelationship:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO relationships (
                    id, source_id, source_type, target_id, target_type, relation_type, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relationship.id,
                    relationship.source_id,
                    relationship.source_type,
                    relationship.target_id,
                    relationship.target_type,
                    relationship.relation_type.value if hasattr(relationship.relation_type, "value") else relationship.relation_type,
                    relationship.created_at.isoformat(),
                    json.dumps(relationship.metadata),
                ),
            )
            await db.commit()
        return relationship

    async def get_relationships_for_node(
        self,
        node_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[MemoryRelationship]:
        clauses = []
        params: List[Any] = []

        if direction == "outgoing":
            clauses.append("source_id = ?")
            params.append(node_id)
        elif direction == "incoming":
            clauses.append("target_id = ?")
            params.append(node_id)
        else:
            clauses.append("(source_id = ? OR target_id = ?)")
            params.extend([node_id, node_id])

        if relation_types:
            placeholders = ",".join("?" for _ in relation_types)
            clauses.append(f"relation_type IN ({placeholders})")
            params.extend([r.value if hasattr(r, "value") else r for r in relation_types])

        query = f"""
            SELECT id, source_id, source_type, target_id, target_type, relation_type, created_at, metadata_json
            FROM relationships WHERE {' AND '.join(clauses)}
        """

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                results = []
                for r in rows:
                    results.append(
                        MemoryRelationship(
                            id=r[0],
                            source_id=r[1],
                            source_type=r[2],
                            target_id=r[3],
                            target_type=r[4],
                            relation_type=RelationType(r[5]),
                            created_at=datetime.fromisoformat(r[6]),
                            metadata=json.loads(r[7]),
                        )
                    )
                return results

    async def get_related_memories(
        self,
        node_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Tuple[MemoryRelationship, Optional[Memory]]]:
        """Resolve each relationship edge to the memory on the *other* end (when that end is a memory node)."""
        relationships = await self.get_relationships_for_node(node_id, direction, relation_types)
        results: List[Tuple[MemoryRelationship, Optional[Memory]]] = []
        for rel in relationships:
            other_id = rel.target_id if rel.source_id == node_id else rel.source_id
            other_type = rel.target_type if rel.source_id == node_id else rel.source_type
            memory = await self.get_memory(other_id) if other_type == "memory" else None
            results.append((rel, memory))
        return results

    async def get_memory_history(self, memory_id: str) -> List[Memory]:
        """Traverse historical lineage through supersedes_memory_id pointers."""
        history: List[Memory] = []
        current_id = memory_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            mem = await self.get_memory(current_id)
            if not mem:
                break
            history.append(mem)
            current_id = mem.supersedes_memory_id

        return history

    async def get_provenance(self, memory_id: str) -> Dict[str, Any]:
        memory = await self.get_memory(memory_id)
        if not memory:
            return {}
        source_event = await self.get_event(memory.source_event_id) if memory.source_event_id else None
        history = await self.get_memory_history(memory_id)
        return {
            "memory": memory.model_dump(mode="json"),
            "source_event": source_event.model_dump(mode="json") if source_event else None,
            "history_chain": [m.model_dump(mode="json") for m in history],
            "provenance_metadata": memory.provenance,
        }
