"""Wires up the singletons the API needs: repository, LLM client, judge, and
the two compiled LangGraph workflows. Built once at app startup."""

import logging

from app.agent.workflow import build_ingest_workflow, build_query_workflow
from app.config import Settings
from app.cost.factory import get_cost_repository
from app.cost.service import CostService
from app.integrations.graph8 import Graph8MCPClient
from app.judgment.factory import get_memory_judge
from app.judgment.interface import BaseMemoryJudge
from app.llm.factory import get_llm_client
from app.llm.interface import BaseLLMClient
from app.memory.factory import get_memory_repository
from app.memory.repository import BaseMemoryRepository
from app.memory.service import DEFAULT_SUBJECT, MemoryService

logger = logging.getLogger("uvicorn.error")


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository: BaseMemoryRepository = get_memory_repository(settings)
        self.llm_client: BaseLLMClient = get_llm_client(settings)
        self.judge: BaseMemoryJudge = get_memory_judge(settings, self.llm_client)
        self.default_subject = DEFAULT_SUBJECT
        self.memory_service = MemoryService(self.repository, self.llm_client, self.judge, self.default_subject)
        self.ingest_workflow = build_ingest_workflow(self.repository, self.llm_client, self.judge, self.default_subject)
        self.query_workflow = build_query_workflow(self.repository, self.llm_client)
        self.cost_service = CostService(get_cost_repository(settings, self.repository))
        self.g8_client = Graph8MCPClient(
            api_key=settings.g8_api_key,
            url=settings.g8_mcp_url,
            allow_mutations=settings.g8_mcp_allow_mutations,
        )
        logger.info(
            "[memtrace.providers] ready storage=%s llm_provider=%s llm_model=%s judge=%s graph8_configured=%s",
            type(self.repository).__name__,
            self.llm_client.provider_name,
            self.llm_client.model_name,
            type(self.judge).__name__,
            bool(settings.g8_api_key),
        )

    @property
    def active_model_name(self) -> str:
        """Whichever model is actually answering, for honest cost-pricing lookups —
        `get_pricing()` falls back to a nonzero default for an unrecognized id, so
        this must reflect the model actually in use, not always `openai_model`."""
        if self.settings.openai_api_key:
            return self.settings.openai_model
        if self.settings.openrouter_api_key:
            return self.settings.openrouter_model
        return "unconfigured"

    async def initialize(self) -> None:
        await self.repository.initialize()
        await self.cost_service.initialize()

    async def close(self) -> None:
        await self.g8_client.close()
        await self.repository.close()
