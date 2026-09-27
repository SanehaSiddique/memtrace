from app.config import Settings
from app.judgment.interface import BaseMemoryJudge
from app.llm.interface import BaseLLMClient


def get_memory_judge(settings: Settings, llm_client: BaseLLMClient) -> BaseMemoryJudge:
    if settings.jev_api_key:
        from app.judgment.jev_live import LiveJEVClient  # imported lazily: optional dependency path

        return LiveJEVClient(api_key=settings.jev_api_key, url=settings.jev_decide_url)
    if settings.typesafe_api_key:
        from app.judgment.jev import JEVMemoryJudge  # imported lazily: optional dependency path

        return JEVMemoryJudge(api_key=settings.typesafe_api_key)
    if settings.ai_gateway_api_key:
        from app.judgment.ai_gateway import AIGatewayJEVClient

        return AIGatewayJEVClient(api_key=settings.ai_gateway_api_key, url=settings.jev_url)

    if settings.typesafe_api_key:
        from app.judgment.jev import JEVMemoryJudge  # imported lazily: optional dependency path

        return JEVMemoryJudge(api_key=settings.typesafe_api_key)

    raise RuntimeError(
        "No live JEV judge is configured. Set AI_GATEWAY_API_KEY for Vercel AI Gateway."
    )
