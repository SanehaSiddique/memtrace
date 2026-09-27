"""Configuration settings for MEMTRACE."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve to the repo-root .env regardless of the process's working directory
# (e.g. `cd backend && uvicorn app.main:app` runs with cwd=backend/).
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # OpenAI / Compatible LLM
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Groq (fast free-tier models; takes priority when configured)
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: Optional[str] = None

    # OpenRouter (free-tier friendly; takes priority over OPENAI_API_KEY when set)
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: Optional[str] = None
    openrouter_site_url: Optional[str] = None
    openrouter_app_name: str = "memtrace"


    # TypeSafe JEV
    typesafe_api_key: Optional[str] = None

    # JEV's own live REST API (jv_live_... key) — confirmed-working production
    # endpoint, takes top priority over every other JEV path below when set.
    jev_api_key: Optional[str] = None
    jev_decide_url: str = "https://jevtypesafeai.com/api/v1/decide"

    # Vercel AI Gateway — JEV bounded memory-lifecycle decisions (real endpoint,
    # takes priority over the OpenRouter-chat JEV substitute when set)
    ai_gateway_api_key: Optional[str] = None
    jev_url: str = "https://ai-gateway.vercel.sh/v1/evaluate"

    # graph8 MCP (external intelligence layer)
    g8_api_key: Optional[str] = None
    g8_mcp_mode: str = "dev"
    g8_mcp_url: str = "https://be.graph8.com/mcp/"
    # The public demo API has no authentication of its own. Keep Graph8 writes,
    # billable operations, and destructive tools disabled unless the operator
    # deliberately opts in after putting suitable access controls in front of it.
    g8_mcp_allow_mutations: bool = False

    # LLM call control. The MVP rule is "stop making multiple unnecessary
    # provider calls", so deterministic repeats are cached and free-tier
    # congestion is backed off from instead of retried into a 429 storm.
    memtrace_llm_cache_enabled: bool = True
    memtrace_llm_cache_ttl_seconds: int = 900
    memtrace_llm_cache_max_entries: int = 512
    openrouter_max_rpm: int = 20  # documented free-model per-minute ceiling
    openrouter_max_backoff_seconds: float = 20.0
    openrouter_model_cooldown_seconds: float = 90.0

    # Comparison demo databases (docs §3, §5.1, §6.1)
    postgres_url: Optional[str] = None
    neo4j_uri: Optional[str] = None
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


    # LangSmith Tracing
    langchain_tracing_v2: str = "false"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "memtrace-hackathon"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Neo4j persistence (required outside tests)
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: str = "neo4j"

    # SQLite is retained only as a lightweight test adapter.
    memtrace_db_path: str = "memtrace.db"
    memtrace_env: str = "development"
    memtrace_default_agent_id: str = "agent-alpha"
    memtrace_debug: bool = True


settings = Settings()
