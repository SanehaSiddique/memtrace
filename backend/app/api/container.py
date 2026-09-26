"""Wires up the singletons the API needs: repository, LLM client, judge, and
the two compiled LangGraph workflows. Built once at app startup."""

from app.agent.workflow import build_ingest_workflow, build_query_workflow
from app.config import Settings
from app.cost.repository import SQLiteCostRepository
from app.cost.service import CostService
from app.judgment.factory import get_memory_judge
from app.judgment.interface import BaseMemoryJudge
from app.llm.factory import get_llm_client
from app.llm.interface import BaseLLMClient
from app.memory.repository import BaseMemoryRepository, SQLiteMemoryRepository
from app.memory.service import DEFAULT_SUBJECT, MemoryService


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository: BaseMemoryRepository = SQLiteMemoryRepository(db_path=settings.memtrace_db_path)
        self.llm_client: BaseLLMClient = get_llm_client(settings)
        self.judge: BaseMemoryJudge = get_memory_judge(settings)
        self.default_subject = DEFAULT_SUBJECT
        self.memory_service = MemoryService(self.repository, self.llm_client, self.judge, self.default_subject)
        self.ingest_workflow = build_ingest_workflow(self.repository, self.llm_client, self.judge, self.default_subject)
        self.query_workflow = build_query_workflow(self.repository, self.llm_client)
        self.cost_service = CostService(SQLiteCostRepository(db_path=settings.memtrace_db_path))

    async def initialize(self) -> None:
        await self.repository.initialize()
        await self.cost_service.initialize()
