"""FastAPI routes. Thin: every handler just adapts HTTP <-> the workflows,
services, and repository already built and tested below the API layer."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from app.api.container import AppContainer
from app.api.schemas import ChatRequest, IngestRequest, QueryRequest, resolve_agent_id
from app.demo.seed import seed_demo_data
from app.evaluation.dataset import EVAL_CASES
from app.evaluation.runner import run_evaluation
from app.memory.models import DebugQueryResult, GraphPathStep, MemoryEvent, MemoryStatus
from app.memory.retrieval import baseline_semantic_retrieve, hybrid_retrieve
from langsmith.run_helpers import get_current_run_tree, traceable

router = APIRouter()


def _container(request: Request) -> AppContainer:
    return request.app.state.container


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/memory")
async def list_memory(request: Request, agent_id: Optional[str] = None, status: Optional[str] = None, limit: int = 200):
    container = _container(request)
    resolved_agent_id = resolve_agent_id(agent_id)
    status_enum = MemoryStatus(status.upper()) if status else None
    memories = await container.repository.list_memories(resolved_agent_id, status=status_enum, limit=limit)
    return [m.model_dump(mode="json") for m in memories]


@router.post("/demo/seed")
async def demo_seed(request: Request):
    container = _container(request)
    records = await seed_demo_data(container.memory_service, agent_id=container.settings.memtrace_default_agent_id)
    return {"operations": [r.model_dump(mode="json") for r in records]}


@router.post("/memory/ingest")
async def ingest(body: IngestRequest, request: Request):
    container = _container(request)
    agent_id = resolve_agent_id(body.agent_id)
    event = MemoryEvent(
        conversation_id=body.conversation_id,
        agent_id=agent_id,
        speaker=body.speaker,
        content=body.content,
        metadata=body.metadata,
    )
    result = await container.ingest_workflow.ainvoke({"event": event})
    return {
        "event_id": event.event_id,
        "operations": [op.model_dump(mode="json") for op in result["operations"]],
    }


@router.post("/agent/chat")
async def chat(body: ChatRequest, request: Request):
    container = _container(request)
    agent_id = resolve_agent_id(body.agent_id)

    event = MemoryEvent(
        conversation_id=body.conversation_id,
        agent_id=agent_id,
        speaker=body.speaker,
        content=body.message,
    )
    ingest_result = await container.ingest_workflow.ainvoke({"event": event})

    query_result = await container.query_workflow.ainvoke(
        {
            "query": body.message,
            "conversation_id": body.conversation_id,
            "agent_id": agent_id,
            "recent_messages": [],
        }
    )

    return {
        "answer": query_result["answer"],
        "memory_operations": [op.model_dump(mode="json") for op in ingest_result["operations"]],
        "selected_memories": [sm.model_dump(mode="json") for sm in query_result["selected_memories"]],
        "excluded_memories": [em.model_dump(mode="json") for em in query_result["excluded_memories"]],
        "trace_metadata": query_result["trace_metadata"],
    }


@router.post("/memory/query")
async def query_memory(body: QueryRequest, request: Request):
    container = _container(request)
    agent_id = resolve_agent_id(body.agent_id)

    result = await container.query_workflow.ainvoke(
        {
            "query": body.query,
            "conversation_id": body.conversation_id,
            "agent_id": agent_id,
            "recent_messages": body.recent_messages,
        }
    )
    return {
        "answer": result["answer"],
        "selected_memories": [sm.model_dump(mode="json") for sm in result["selected_memories"]],
        "excluded_memories": [em.model_dump(mode="json") for em in result["excluded_memories"]],
        "context": result["context"].model_dump(mode="json"),
        "trace_metadata": result["trace_metadata"],
    }


@router.get("/memory/{memory_id}")
async def get_memory(memory_id: str, request: Request):
    memory = await _container(request).repository.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory.model_dump(mode="json")


@router.get("/memory/{memory_id}/history")
async def get_memory_history(memory_id: str, request: Request):
    history = await _container(request).repository.get_memory_history(memory_id)
    if not history:
        raise HTTPException(status_code=404, detail="Memory not found")
    return [m.model_dump(mode="json") for m in history]


@router.get("/memory/{memory_id}/graph")
async def get_memory_graph(memory_id: str, request: Request):
    repository = _container(request).repository
    memory = await repository.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    related = await repository.get_related_memories(memory_id, direction="both")
    return {
        "memory": memory.model_dump(mode="json"),
        "relationships": [
            {
                "relationship": rel.model_dump(mode="json"),
                "related_memory": other.model_dump(mode="json") if other else None,
            }
            for rel, other in related
        ],
    }


@router.get("/memory/{memory_id}/provenance")
async def get_memory_provenance(memory_id: str, request: Request):
    provenance = await _container(request).repository.get_provenance(memory_id)
    if not provenance:
        raise HTTPException(status_code=404, detail="Memory not found")
    return provenance


async def _graph_paths_for(repository, memory_ids: List[str]) -> List[List[GraphPathStep]]:
    paths: List[List[GraphPathStep]] = []
    for memory_id in memory_ids:
        for rel, other in await repository.get_related_memories(memory_id, direction="both"):
            label = f"{other.subject} {other.predicate} {other.object}" if other else rel.target_id
            paths.append([GraphPathStep(relationship=rel, node_id=memory_id, node_label=label)])
    return paths


@router.post("/debug/query", response_model=DebugQueryResult)
async def debug_query(body: QueryRequest, request: Request):
    container = _container(request)
    agent_id = resolve_agent_id(body.agent_id)

    @traceable(name="debug.query")
    async def _run():
        return await container.query_workflow.ainvoke(
            {
                "query": body.query,
                "conversation_id": body.conversation_id,
                "agent_id": agent_id,
                "recent_messages": body.recent_messages,
            }
        )

    result = await _run()
    run_tree = get_current_run_tree()
    memory_ids = [sm.memory.id for sm in result["selected_memories"]]
    graph_paths = await _graph_paths_for(container.repository, memory_ids)

    return DebugQueryResult(
        query=body.query,
        conversation_id=body.conversation_id,
        agent_id=agent_id,
        answer=result["answer"],
        retrieved=result["retrieved_memories"],
        excluded=result["excluded_memories"],
        graph_paths=graph_paths,
        context=result["context"],
        langsmith_run_id=str(run_tree.id) if run_tree else None,
        trace_metadata=result["trace_metadata"],
    )


@router.post("/debug/replay")
async def debug_replay(body: QueryRequest, request: Request):
    container = _container(request)
    agent_id = resolve_agent_id(body.agent_id)

    baseline = await baseline_semantic_retrieve(container.repository, container.llm_client, body.query, agent_id)
    memtrace = await hybrid_retrieve(container.repository, container.llm_client, body.query, agent_id)

    query_result = await container.query_workflow.ainvoke(
        {
            "query": body.query,
            "conversation_id": body.conversation_id,
            "agent_id": agent_id,
            "recent_messages": body.recent_messages,
        }
    )

    return {
        "query": body.query,
        "baseline": {
            "description": "flat semantic-only top-k, no graph traversal, no temporal filtering",
            "retrieved": [sm.model_dump(mode="json") for sm in baseline],
        },
        "memtrace": {
            "description": "hybrid retrieval + graph traversal + temporal filtering",
            "retrieved": [sm.model_dump(mode="json") for sm in memtrace],
            "selected": [sm.model_dump(mode="json") for sm in query_result["selected_memories"]],
            "excluded": [em.model_dump(mode="json") for em in query_result["excluded_memories"]],
            "answer": query_result["answer"],
        },
    }


@router.post("/evaluate")
async def evaluate(request: Request):
    container = _container(request)
    agent_id = container.settings.memtrace_default_agent_id
    report = await run_evaluation(container.repository, container.llm_client, agent_id, EVAL_CASES)
    return report.model_dump(mode="json")
