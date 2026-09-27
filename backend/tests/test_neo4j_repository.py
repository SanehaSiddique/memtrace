"""Unit checks for Neo4j mapping and query behavior without a live database."""

import pytest

from app.memory.models import Memory, MemoryRelationship, RelationType
from app.memory.neo4j_repository import Neo4jMemoryRepository


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.verified = None
        self.closed = False

    async def verify_connectivity(self, **config):
        self.verified = config

    async def execute_query(self, cypher, parameters_=None, database_=None):
        self.calls.append((cypher, parameters_ or {}, database_))
        if "RETURN r" in cypher:
            return ([{"r": {}}], None, None)
        return ([], None, None)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_neo4j_repository_initializes_and_writes_native_relationships():
    driver = FakeDriver()
    repository = Neo4jMemoryRepository("neo4j://unused", "neo4j", "secret", driver=driver)

    await repository.initialize()
    memory = Memory(id="mem_a", subject="User", predicate="prefers", object="tea", content="User prefers tea")
    await repository.create_memory(memory, [0.1, 0.2])
    await repository.add_relationship(
        MemoryRelationship(
            id="rel_a",
            source_id="mem_a",
            target_id="mem_a",
            relation_type=RelationType.RELATED_TO,
        )
    )
    await repository.close()

    assert driver.verified == {"database": "neo4j"}
    assert any("CREATE (m:Memory)" in query for query, _, _ in driver.calls)
    assert any("[r:RELATED_TO" in query for query, _, _ in driver.calls)
    assert all(database == "neo4j" for _, _, database in driver.calls)
    assert driver.closed is True


def test_neo4j_memory_serialization_round_trip():
    original = Memory(subject="User", predicate="timezone", object="Asia/Karachi", content="User is in Asia/Karachi")
    restored = Neo4jMemoryRepository._to_memory(Neo4jMemoryRepository._memory_props(original))
    assert restored == original
