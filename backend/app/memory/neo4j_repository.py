"""Neo4j-backed memory and relationship repository.

Memories and source events are stored as native Neo4j nodes. Memory lifecycle
links are stored using their real relationship types (for example
``REPLACED_BY`` and ``SUPERSEDES``), so the same graph powers retrieval,
provenance APIs, Neo4j Browser, and the frontend visualization.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from neo4j import AsyncGraphDatabase

from app.memory.models import Memory, MemoryEvent, MemoryRelationship, MemoryStatus, RelationType
from app.memory.repository import BaseMemoryRepository, _cosine_similarity

logger = logging.getLogger("uvicorn.error")

_NODE_LABELS = {"memory": "Memory", "event": "MemoryEvent", "entity": "Entity"}


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


class Neo4jMemoryRepository(BaseMemoryRepository):
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j", driver=None) -> None:
        self.database = database
        self.driver = driver or AsyncGraphDatabase.driver(uri, auth=(username, password))

    async def _query(self, cypher: str, **parameters):
        records, _, _ = await self.driver.execute_query(
            cypher,
            parameters_=parameters,
            database_=self.database,
        )
        return records

    async def initialize(self) -> None:
        await self.driver.verify_connectivity(database=self.database)
        statements = (
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT memory_event_id IF NOT EXISTS FOR (e:MemoryEvent) REQUIRE e.event_id IS UNIQUE",
            "CREATE INDEX memory_agent_status IF NOT EXISTS FOR (m:Memory) ON (m.agent_id, m.status)",
            "CREATE INDEX memory_subject_predicate IF NOT EXISTS FOR (m:Memory) ON (m.agent_id, m.subject, m.predicate)",
        )
        for statement in statements:
            await self._query(statement)
        logger.info("[memtrace.neo4j] connected database=%s", self.database)

    async def close(self) -> None:
        await self.driver.close()

    @staticmethod
    def _event_props(event: MemoryEvent) -> dict:
        return {
            "event_id": event.event_id,
            "conversation_id": event.conversation_id,
            "agent_id": event.agent_id,
            "timestamp": event.timestamp.isoformat(),
            "speaker": event.speaker,
            "content": event.content,
            "metadata_json": json.dumps(event.metadata),
        }

    @staticmethod
    def _memory_props(memory: Memory) -> dict:
        return {
            "id": memory.id,
            "agent_id": memory.agent_id,
            "memory_type": _enum_value(memory.memory_type),
            "subject": memory.subject,
            "predicate": memory.predicate,
            "object": memory.object,
            "content": memory.content,
            "confidence": memory.confidence,
            "status": _enum_value(memory.status),
            "created_at": memory.created_at.isoformat(),
            "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
            "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
            "supersedes_memory_id": memory.supersedes_memory_id,
            "superseded_by_memory_id": memory.superseded_by_memory_id,
            "source_event_id": memory.source_event_id,
            "provenance_json": json.dumps(memory.provenance),
            "metadata_json": json.dumps(memory.metadata),
        }

    @staticmethod
    def _to_event(node) -> MemoryEvent:
        data = dict(node)
        return MemoryEvent(
            event_id=data["event_id"],
            conversation_id=data["conversation_id"],
            agent_id=data["agent_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            speaker=data["speaker"],
            content=data["content"],
            metadata=json.loads(data.get("metadata_json") or "{}"),
        )

    @staticmethod
    def _to_memory(node) -> Memory:
        data = dict(node)
        return Memory(
            id=data["id"],
            agent_id=data["agent_id"],
            memory_type=data["memory_type"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            content=data["content"],
            confidence=data["confidence"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            supersedes_memory_id=data.get("supersedes_memory_id"),
            superseded_by_memory_id=data.get("superseded_by_memory_id"),
            source_event_id=data.get("source_event_id"),
            provenance=json.loads(data.get("provenance_json") or "{}"),
            metadata=json.loads(data.get("metadata_json") or "{}"),
        )

    @staticmethod
    def _to_relationship(data: dict) -> MemoryRelationship:
        return MemoryRelationship(
            id=data["id"],
            source_id=data["source_id"],
            source_type=data["source_type"],
            target_id=data["target_id"],
            target_type=data["target_type"],
            relation_type=data["relation_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=json.loads(data.get("metadata_json") or "{}"),
        )

    async def save_event(self, event: MemoryEvent) -> MemoryEvent:
        await self._query(
            "MERGE (e:MemoryEvent {event_id: $event_id}) SET e = $props",
            event_id=event.event_id,
            props=self._event_props(event),
        )
        return event

    async def get_event(self, event_id: str) -> Optional[MemoryEvent]:
        records = await self._query("MATCH (e:MemoryEvent {event_id: $event_id}) RETURN e", event_id=event_id)
        return self._to_event(records[0]["e"]) if records else None

    async def create_memory(self, memory: Memory, embedding: Optional[List[float]] = None) -> Memory:
        props = self._memory_props(memory)
        props["embedding"] = embedding
        await self._query("CREATE (m:Memory) SET m = $props", props=props)
        return memory

    async def update_memory(self, memory: Memory) -> Memory:
        await self._query(
            "MATCH (m:Memory {id: $id}) SET m += $props",
            id=memory.id,
            props=self._memory_props(memory),
        )
        return memory

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        records = await self._query("MATCH (m:Memory {id: $id}) RETURN m", id=memory_id)
        return self._to_memory(records[0]["m"]) if records else None

    async def list_memories(
        self,
        agent_id: str,
        status: Optional[MemoryStatus] = None,
        subject: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        clauses = ["m.agent_id = $agent_id"]
        if status:
            clauses.append("m.status = $status")
        if subject:
            clauses.append("toLower(m.subject) = toLower($subject)")
        records = await self._query(
            f"MATCH (m:Memory) WHERE {' AND '.join(clauses)} RETURN m ORDER BY m.created_at DESC LIMIT $limit",
            agent_id=agent_id,
            status=_enum_value(status) if status else None,
            subject=subject,
            limit=max(1, int(limit)),
        )
        return [self._to_memory(record["m"]) for record in records]

    async def get_active_memories(self, agent_id: str, subject: Optional[str] = None) -> List[Memory]:
        return await self.list_memories(agent_id, MemoryStatus.ACTIVE, subject, 1000)

    async def get_historical_memories(self, agent_id: str, subject: Optional[str] = None) -> List[Memory]:
        return await self.list_memories(agent_id, MemoryStatus.HISTORICAL, subject, 1000)

    async def archive_memory(self, memory_id: str, reason: str = "") -> Optional[Memory]:
        memory = await self.get_memory(memory_id)
        if memory is None:
            return None
        memory.status = MemoryStatus.ARCHIVED
        memory.valid_until = memory.valid_until or datetime.now(timezone.utc)
        if reason:
            memory.metadata = {**memory.metadata, "archive_reason": reason}
        return await self.update_memory(memory)

    async def delete_memory(self, memory_id: str, reason: str = "") -> Optional[Memory]:
        memory = await self.get_memory(memory_id)
        if memory is None:
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
        status_clause = " AND m.status = $status" if status else ""
        records = await self._query(
            "MATCH (m:Memory) "
            "WHERE m.agent_id = $agent_id AND toLower(m.subject) = toLower($subject) "
            f"AND toLower(m.predicate) = toLower($predicate){status_clause} "
            "RETURN m ORDER BY m.created_at DESC",
            agent_id=agent_id,
            subject=subject,
            predicate=predicate,
            status=_enum_value(status) if status else None,
        )
        return [self._to_memory(record["m"]) for record in records]

    async def search_semantic(
        self,
        agent_id: str,
        query_embedding: List[float],
        limit: int = 10,
        status: Optional[MemoryStatus] = None,
    ) -> List[Tuple[Memory, float]]:
        status_clause = " AND m.status = $status" if status else ""
        records = await self._query(
            f"MATCH (m:Memory) WHERE m.agent_id = $agent_id AND m.embedding IS NOT NULL{status_clause} "
            "RETURN m, m.embedding AS embedding",
            agent_id=agent_id,
            status=_enum_value(status) if status else None,
        )
        scored = [
            (self._to_memory(record["m"]), _cosine_similarity(query_embedding, list(record["embedding"])))
            for record in records
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, int(limit))]

    async def add_relationship(self, relationship: MemoryRelationship) -> MemoryRelationship:
        source_label = _NODE_LABELS.get(relationship.source_type)
        target_label = _NODE_LABELS.get(relationship.target_type)
        if source_label is None or target_label is None:
            raise ValueError("Relationship endpoints must be memory, event, or entity nodes")
        source_key = "event_id" if relationship.source_type == "event" else "id"
        target_key = "event_id" if relationship.target_type == "event" else "id"
        relation_type = _enum_value(relationship.relation_type)
        if relation_type not in {item.value for item in RelationType}:
            raise ValueError(f"Unsupported relationship type: {relation_type}")
        records = await self._query(
            f"MATCH (source:{source_label} {{{source_key}: $source_id}}) "
            f"MATCH (target:{target_label} {{{target_key}: $target_id}}) "
            f"MERGE (source)-[r:{relation_type} {{id: $id}}]->(target) "
            "SET r.source_id = $source_id, r.source_type = $source_type, "
            "r.target_id = $target_id, r.target_type = $target_type, "
            "r.relation_type = $relation_type, r.created_at = $created_at, r.metadata_json = $metadata_json "
            "RETURN r",
            id=relationship.id,
            source_id=relationship.source_id,
            source_type=relationship.source_type,
            target_id=relationship.target_id,
            target_type=relationship.target_type,
            relation_type=relation_type,
            created_at=relationship.created_at.isoformat(),
            metadata_json=json.dumps(relationship.metadata),
        )
        if not records:
            raise ValueError(
                f"Cannot create relationship {relationship.id}: source or target node does not exist"
            )
        return relationship

    async def get_relationships_for_node(
        self,
        node_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[MemoryRelationship]:
        direction_clause = {
            "outgoing": "r.source_id = $node_id",
            "incoming": "r.target_id = $node_id",
            "both": "(r.source_id = $node_id OR r.target_id = $node_id)",
        }.get(direction)
        if direction_clause is None:
            raise ValueError("direction must be outgoing, incoming, or both")
        type_values = [_enum_value(item) for item in relation_types] if relation_types else None
        records = await self._query(
            "MATCH ()-[r]->() "
            f"WHERE r.id IS NOT NULL AND {direction_clause} "
            "AND ($relation_types IS NULL OR r.relation_type IN $relation_types) "
            "RETURN properties(r) AS relationship ORDER BY r.created_at",
            node_id=node_id,
            relation_types=type_values,
        )
        return [self._to_relationship(dict(record["relationship"])) for record in records]

    async def get_related_memories(
        self,
        node_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Tuple[MemoryRelationship, Optional[Memory]]]:
        relationships = await self.get_relationships_for_node(node_id, direction, relation_types)
        results = []
        for relationship in relationships:
            other_id = relationship.target_id if relationship.source_id == node_id else relationship.source_id
            other_type = relationship.target_type if relationship.source_id == node_id else relationship.source_type
            other = await self.get_memory(other_id) if other_type == "memory" else None
            results.append((relationship, other))
        return results

    async def get_memory_history(self, memory_id: str) -> List[Memory]:
        history = []
        current_id = memory_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            memory = await self.get_memory(current_id)
            if memory is None:
                break
            history.append(memory)
            current_id = memory.supersedes_memory_id
        return history

    async def get_provenance(self, memory_id: str) -> Dict[str, Any]:
        memory = await self.get_memory(memory_id)
        if memory is None:
            return {}
        event = await self.get_event(memory.source_event_id) if memory.source_event_id else None
        history = await self.get_memory_history(memory_id)
        return {
            "memory": memory.model_dump(mode="json"),
            "source_event": event.model_dump(mode="json") if event else None,
            "history_chain": [item.model_dump(mode="json") for item in history],
            "provenance_metadata": memory.provenance,
        }
