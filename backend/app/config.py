"""Configuration settings for MEMTRACE."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI / Compatible LLM
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # TypeSafe JEV
    typesafe_api_key: Optional[str] = None

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
