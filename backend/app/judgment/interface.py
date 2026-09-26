"""MemoryJudge interface: the single bounded decision point for memory lifecycle.

Given a freshly extracted candidate fact and whatever active memories already
exist for the same (subject, predicate), a MemoryJudge decides ADD / UPDATE /
MERGE / ARCHIVE / DELETE / REVIEW / NOOP. Implementations must NOT act as a
general-purpose chat agent — this is a narrow, structured judgment, which is
exactly what TypeSafe's JEV ("System One") API is built for.
"""

from abc import ABC, abstractmethod
from typing import List

from app.memory.models import CandidateMemory, Memory, MemoryEvent, MemoryJudgment


class BaseMemoryJudge(ABC):
    @abstractmethod
    async def judge(
        self,
        candidate: CandidateMemory,
        existing_active: List[Memory],
        event: MemoryEvent,
    ) -> MemoryJudgment: ...
