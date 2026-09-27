"""LangGraph node functions.

Each node is small, independently callable, and traced under the run names
called out in the project brief (memory.retrieve, memory.graph_traverse, ...).
Nodes depend only on `BaseMemoryRepository` / `BaseLLMClient` / `BaseMemoryJudge`
interfaces, injected via closures from `app.agent.workflow`, never on a
concrete database or provider.
"""

from app.context.builder import build_context
from app.judgment.interface import BaseMemoryJudge
from app.llm.interface import BaseLLMClient
from app.memory.consolidation import apply_operation
from app.memory.extraction import extract_candidates
from app.memory.repository import BaseMemoryRepository
from app.memory.retrieval import (
    apply_temporal_filter,
    find_linkable_active_memories,
    retrieve_semantic,
    score_and_rank,
    traverse_graph,
    wants_historical_context,
)
from app.tracing.langsmith import traced

_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's query using ONLY the memory listed in the context. "
    "If the context says a fact is historical or excluded, do not present it as current. "
    "Be concise."
)


# ---------------------------------------------------------------------------
# Query-time workflow nodes
# ---------------------------------------------------------------------------


def make_analyze_query_node():
    @traced(name="query.analyze")
    async def analyze_query(state: dict) -> dict:
        query = state["query"]
        return {"wants_history": wants_historical_context(query), "errors": []}

    return analyze_query


def make_retrieve_memory_node(repository: BaseMemoryRepository, llm_client: BaseLLMClient):
    @traced(name="memory.retrieve")
    async def retrieve_memory(state: dict) -> dict:
        candidates = await retrieve_semantic(repository, llm_client, state["query"], state["agent_id"])
        return {"candidate_pool": candidates}

    return retrieve_memory


def make_traverse_graph_node(repository: BaseMemoryRepository):
    @traced(name="memory.graph_traverse")
    async def traverse_graph_node(state: dict) -> dict:
        expanded = await traverse_graph(repository, state["candidate_pool"])
        ranked = score_and_rank(expanded, top_k=8)
        return {"retrieved_memories": ranked}

    return traverse_graph_node


def make_filter_temporal_node():
    @traced(name="memory.filter_temporal")
    async def filter_temporal(state: dict) -> dict:
        selected, excluded = apply_temporal_filter(state["query"], state["retrieved_memories"])
        return {"selected_memories": selected, "excluded_memories": excluded}

    return filter_temporal


def make_build_context_node():
    @traced(name="context.build")
    async def build_context_node(state: dict) -> dict:
        context = build_context(
            state["query"],
            state["selected_memories"],
            state["excluded_memories"],
            recent_messages=state.get("recent_messages"),
        )
        return {"context": context}

    return build_context_node


def make_generate_response_node(llm_client: BaseLLMClient):
    @traced(name="agent.answer")
    async def generate_response(state: dict) -> dict:
        context = state["context"]
        answer = await llm_client.chat(_ANSWER_SYSTEM_PROMPT, context.context_text)

        trace_metadata = {
            "retrieval_mode": "hybrid",
            "selected_memory_count": len(state["selected_memories"]),
            "excluded_memory_count": len(state["excluded_memories"]),
            "agent_id": state["agent_id"],
            "conversation_id": state.get("conversation_id"),
            "memory_ids": [sm.memory.id for sm in state["selected_memories"]],
        }
        return {"answer": answer, "trace_metadata": trace_metadata}

    return generate_response


# ---------------------------------------------------------------------------
# Ingest-time workflow nodes
# ---------------------------------------------------------------------------


def make_ingest_event_node(repository: BaseMemoryRepository):
    @traced(name="memory.ingest_event")
    async def ingest_event(state: dict) -> dict:
        await repository.save_event(state["event"])
        return {"errors": []}

    return ingest_event


def make_extract_memory_node(llm_client: BaseLLMClient, default_subject: str):
    @traced(name="memory.extract")
    async def extract_memory(state: dict) -> dict:
        candidates = await extract_candidates(state["event"], default_subject, llm_client)
        return {"candidates": candidates}

    return extract_memory


def make_judge_and_apply_node(repository: BaseMemoryRepository, llm_client: BaseLLMClient, judge: BaseMemoryJudge):
    @traced(name="memory.judge")
    async def judge_and_apply(state: dict) -> dict:
        event = state["event"]
        operations = []
        for candidate in state["candidates"]:
            existing_active = await find_linkable_active_memories(
                repository,
                llm_client,
                event.agent_id,
                candidate.subject,
                candidate.predicate,
                candidate.object,
                candidate.content,
            )
            judgment = await judge.judge(candidate, existing_active, event)
            record = await apply_operation(repository, llm_client, judgment, candidate, event)
            operations.append(record)
        return {"operations": operations}

    return judge_and_apply
