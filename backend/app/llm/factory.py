from app.config import Settings
from app.llm.counted import CountedLLMClient
from app.llm.groq_client import GroqLLMClient
from app.llm.interface import BaseLLMClient
from app.llm.mock import MockLLMClient
from app.llm.openai_client import OpenAICompatibleLLMClient
from app.llm.openrouter_client import OpenRouterLLMClient


def _wrap_live_client(client: BaseLLMClient, settings: Settings) -> BaseLLMClient:
    return CountedLLMClient(
        client,
        cache_enabled=settings.memtrace_llm_cache_enabled,
        chat_cache_ttl_seconds=settings.memtrace_llm_cache_ttl_seconds,
        chat_cache_max_entries=settings.memtrace_llm_cache_max_entries,
    )


def get_llm_client(settings: Settings) -> BaseLLMClient:
    if settings.groq_api_key:
        return _wrap_live_client(
            GroqLLMClient(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                base_url=settings.groq_base_url,
            ),
            settings,
        )
    if settings.openrouter_api_key:
        return _wrap_live_client(
            OpenRouterLLMClient(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                base_url=settings.openrouter_base_url,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
            ),
            settings,
        )
    if settings.openai_api_key:
        return _wrap_live_client(
            OpenAICompatibleLLMClient(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.openai_model,
            ),
            settings,
        )
    return MockLLMClient()

