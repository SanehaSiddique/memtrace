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

    # OpenRouter is an optional live fallback when OPENAI_API_KEY is absent.
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: Optional[str] = None
    openrouter_site_url: Optional[str] = None
    openrouter_app_name: str = "memtrace"

    # TypeSafe JEV
    typesafe_api_key: Optional[str] = None

    # Vercel AI Gateway — preferred JEV provider for bounded memory-lifecycle
    # decisions.
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

    # LangSmith Tracing
    langchain_tracing_v2: str = "false"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "memtrace-hackathon"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Persistence
    memtrace_db_path: str = "memtrace.db"
    memtrace_env: str = "development"
    memtrace_default_agent_id: str = "agent-alpha"
    memtrace_debug: bool = True


settings = Settings()
