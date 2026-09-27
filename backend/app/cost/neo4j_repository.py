"""Neo4j persistence for run costs and memory-level cost impacts."""

from datetime import datetime
from typing import List

from app.cost.models import MemoryCostImpact, RunRecord


class Neo4jCostRepository:
    def __init__(self, driver, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database

    async def _query(self, cypher: str, **parameters):
        records, _, _ = await self.driver.execute_query(
            cypher, parameters_=parameters, database_=self.database
        )
        return records

    async def initialize(self) -> None:
        for statement in (
            "CREATE CONSTRAINT run_cost_id IF NOT EXISTS FOR (r:RunCost) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT memory_cost_impact_id IF NOT EXISTS FOR (i:MemoryCostImpact) REQUIRE i.id IS UNIQUE",
            "CREATE INDEX run_cost_agent IF NOT EXISTS FOR (r:RunCost) ON (r.agent_id, r.created_at)",
            "CREATE INDEX memory_impact_agent IF NOT EXISTS FOR (i:MemoryCostImpact) ON (i.agent_id, i.created_at)",
        ):
            await self._query(statement)

    @staticmethod
    def _props(model) -> dict:
        data = model.model_dump()
        data["created_at"] = model.created_at.isoformat()
        return data

    @staticmethod
    def _run(node) -> RunRecord:
        data = dict(node)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return RunRecord(**data)

    @staticmethod
    def _impact(node) -> MemoryCostImpact:
        data = dict(node)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return MemoryCostImpact(**data)

    async def record(self, run: RunRecord) -> RunRecord:
        await self._query("CREATE (r:RunCost) SET r = $props", props=self._props(run))
        return run

    async def list_all(self, agent_id: str, limit: int = 100000) -> List[RunRecord]:
        records = await self._query(
            "MATCH (r:RunCost {agent_id: $agent_id}) RETURN r ORDER BY r.created_at ASC LIMIT $limit",
            agent_id=agent_id,
            limit=max(1, int(limit)),
        )
        return [self._run(record["r"]) for record in records]

    async def list_recent(self, agent_id: str, limit: int = 20) -> List[RunRecord]:
        records = await self._query(
            "MATCH (r:RunCost {agent_id: $agent_id}) RETURN r ORDER BY r.created_at DESC LIMIT $limit",
            agent_id=agent_id,
            limit=max(1, int(limit)),
        )
        return [self._run(record["r"]) for record in records]

    async def record_impact(self, impact: MemoryCostImpact) -> MemoryCostImpact:
        await self._query(
            "CREATE (i:MemoryCostImpact) SET i = $props "
            "WITH i OPTIONAL MATCH (m:Memory {id: i.memory_id}) "
            "FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END | MERGE (i)-[:IMPACTS]->(m)) "
            "WITH i OPTIONAL MATCH (r:RunCost {id: i.run_id}) "
            "FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END | MERGE (i)-[:FROM_RUN]->(r))",
            props=self._props(impact),
        )
        return impact

    async def list_impacts(self, agent_id: str, limit: int = 100000) -> List[MemoryCostImpact]:
        records = await self._query(
            "MATCH (i:MemoryCostImpact {agent_id: $agent_id}) RETURN i ORDER BY i.created_at ASC LIMIT $limit",
            agent_id=agent_id,
            limit=max(1, int(limit)),
        )
        return [self._impact(record["i"]) for record in records]

    async def list_impacts_for_memory(self, memory_id: str) -> List[MemoryCostImpact]:
        records = await self._query(
            "MATCH (i:MemoryCostImpact {memory_id: $memory_id}) RETURN i ORDER BY i.created_at ASC",
            memory_id=memory_id,
        )
        return [self._impact(record["i"]) for record in records]
