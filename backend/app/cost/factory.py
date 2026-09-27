"""Select a cost repository matching the configured memory backend."""

from app.config import Settings
from app.cost.neo4j_repository import Neo4jCostRepository
from app.cost.repository import SQLiteCostRepository
from app.memory.neo4j_repository import Neo4jMemoryRepository
from app.memory.repository import BaseMemoryRepository


def get_cost_repository(settings: Settings, memory_repository: BaseMemoryRepository):
    if isinstance(memory_repository, Neo4jMemoryRepository):
        return Neo4jCostRepository(memory_repository.driver, memory_repository.database)
    return SQLiteCostRepository(settings.memtrace_db_path)
