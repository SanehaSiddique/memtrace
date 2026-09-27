"""Request/response bodies for the FastAPI layer. Kept separate from the
domain models in app.memory.models, which describe the memory graph itself."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import settings


class IngestRequest(BaseModel):
    content: str
    conversation_id: str = "default_conversation"
    agent_id: Optional[str] = None
    speaker: str = "user"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default_conversation"
    agent_id: Optional[str] = None
    speaker: str = "user"
    remember: bool = False
    recent_messages: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    conversation_id: str = "default_conversation"
    agent_id: Optional[str] = None
    recent_messages: List[str] = Field(default_factory=list)


class Graph8ToolCallRequest(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)


def resolve_agent_id(agent_id: Optional[str]) -> str:
    return agent_id or settings.memtrace_default_agent_id
