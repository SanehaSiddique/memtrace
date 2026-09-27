"""Wires up the singletons the API needs: legacy workflows, memory stores,
LLM clients, Graph8 MCP client, JEV client, and the dual-agent comparison
orchestrator. Built once at app startup."""

from typing import Optional

from app.agent.workflow import build_ingest_workflow, build_query_workflow
from app.agent1.memory import Agent1Memory
from app.agent2.memory import Agent2Memory
from app.config import Settings
from app.core.graph8_client import Graph8MCPClient
from app.core.jev_client import JEVClient
from app.core.llm_client import LLMClient
from app.cost.repository import SQLiteCostRepository
from app.cost.service import CostService
from app.db.neo4j_driver import FactGraphStore, open_fact_graph
from app.db.postgres import FactsStore, open_facts_store
from app.judgment.factory import get_memory_judge
from app.judgment.interface import BaseMemoryJudge
from app.llm.factory import get_llm_client
from app.llm.interface import BaseLLMClient
from app.memory.repository import BaseMemoryRepository, SQLiteMemoryRepository
from app.memory.service import DEFAULT_SUBJECT, MemoryService
from app.metrics.collector import SessionMetricsStore
from app.orchestrator import ComparisonOrchestrator


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Legacy single-agent workflow dependencies
        self.repository: BaseMemoryRepository = SQLiteMemoryRepository(db_path=settings.memtrace_db_path)
        self.llm_client: BaseLLMClient = get_llm_client(settings)
        self.judge: BaseMemoryJudge = get_memory_judge(settings, self.llm_client)
        self.default_subject = DEFAULT_SUBJECT
        self.memory_service = MemoryService(self.repository, self.llm_client, self.judge, self.default_subject)
        self.ingest_workflow = build_ingest_workflow(self.repository, self.llm_client, self.judge, self.default_subject)
        self.query_workflow = build_query_workflow(self.repository, self.llm_client)
        self.cost_service = CostService(SQLiteCostRepository(db_path=settings.memtrace_db_path))

        # Core clients for dual-agent benchmarking (§4)
        self.core_llm = LLMClient(
            providers=[self.llm_client],
            cache_enabled=settings.memtrace_llm_cache_enabled,
            cache_ttl_seconds=settings.memtrace_llm_cache_ttl_seconds,
            cache_max_entries=settings.memtrace_llm_cache_max_entries,
        )
        self.tools_client = Graph8MCPClient(
            api_key=settings.g8_api_key,
            mode=settings.g8_mcp_mode,
        )
        self.jev_client = JEVClient(
            api_key=settings.ai_gateway_api_key or settings.typesafe_api_key,
            url=settings.jev_url,
        )

        # Stores & Agents
        self.facts_store: Optional[FactsStore] = None
        self.fact_graph: Optional[FactGraphStore] = None
        self.agent1_memory: Optional[Agent1Memory] = None
        self.agent2_memory: Optional[Agent2Memory] = None
        self.metrics_store = SessionMetricsStore()
        self.orchestrator: Optional[ComparisonOrchestrator] = None

    @property
    def active_model_name(self) -> str:
        return self.llm_client.model_name

    async def initialize(self) -> None:
        await self.repository.initialize()
        await self.cost_service.initialize()

        # Connect databases (real or transparent memory fallbacks)
        self.facts_store = await open_facts_store(self.settings.postgres_url)
        self.fact_graph = await open_fact_graph(
            self.settings.neo4j_uri,
            self.settings.neo4j_user,
            self.settings.neo4j_password,
        )

        # Connect MCP tool client
        try:
            await self.tools_client.connect()
        except Exception:
            pass  # tools_client gracefully degrades if server not running

        # Initialize Agent memories & comparison orchestrator
        self.agent1_memory = Agent1Memory(facts_store=self.facts_store)
        self.agent2_memory = Agent2Memory(graph_store=self.fact_graph)

        self.orchestrator = ComparisonOrchestrator(
            llm=self.core_llm,
            tools=self.tools_client,
            jev=self.jev_client,
            agent1_memory=self.agent1_memory,
            agent2_memory=self.agent2_memory,
            metrics_store=self.metrics_store,
        )

    async def aclose(self) -> None:
        if self.tools_client:
            await self.tools_client.aclose()
        if self.facts_store:
            await self.facts_store.aclose()
        if self.fact_graph:
            await self.fact_graph.aclose()
