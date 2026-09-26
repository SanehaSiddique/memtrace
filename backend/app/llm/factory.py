from app.config import Settings
from app.llm.interface import BaseLLMClient
from app.llm.mock import MockLLMClient
from app.llm.openai_client import OpenAICompatibleLLMClient


def get_llm_client(settings: Settings) -> BaseLLMClient:
    if settings.openai_api_key:
        return OpenAICompatibleLLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    return MockLLMClient()
