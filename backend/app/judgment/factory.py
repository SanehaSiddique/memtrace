from app.config import Settings
from app.judgment.interface import BaseMemoryJudge
from app.judgment.mock import MockMemoryJudge


def get_memory_judge(settings: Settings) -> BaseMemoryJudge:
    if settings.typesafe_api_key:
        from app.judgment.jev import JEVMemoryJudge  # imported lazily: optional dependency path

        return JEVMemoryJudge(api_key=settings.typesafe_api_key)
    return MockMemoryJudge()
