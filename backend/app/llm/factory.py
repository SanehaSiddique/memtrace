from app.config import Settings
from app.llm.interface import BaseLLMClient
from app.llm.openai_client import OpenAICompatibleLLMClient
from app.llm.openrouter_client import OpenRouterLLMClient


def get_llm_client(settings: Settings) -> BaseLLMClient:
    if settings.openai_api_key:
        return OpenAICompatibleLLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    if settings.openrouter_api_key:
        return OpenRouterLLMClient(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )

    raise RuntimeError(
        "No live LLM is configured. Set OPENAI_API_KEY (preferred) or OPENROUTER_API_KEY."
    )
