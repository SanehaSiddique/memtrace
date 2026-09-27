from app.config import Settings
from app.memory.neo4j_repository import Neo4jMemoryRepository
from app.memory.repository import BaseMemoryRepository


def get_memory_repository(settings: Settings) -> BaseMemoryRepository:
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", settings.neo4j_uri),
            ("NEO4J_USERNAME", settings.neo4j_username),
            ("NEO4J_PASSWORD", settings.neo4j_password),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Neo4j is required. Set {', '.join(missing)} in .env.")
    return Neo4jMemoryRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
