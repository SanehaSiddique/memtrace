"""Wires the small node functions in app.agent.nodes into two LangGraph graphs:

  build_ingest_workflow() : EVENT -> extraction -> judgment -> lifecycle operation
  build_query_workflow()  : QUERY -> hybrid retrieval -> temporal filter -> context -> answer

Long-term memory lives in `BaseMemoryRepository`, injected into each node via
closure; LangGraph only ever holds per-run scratch state (`QueryState`/`IngestState`).
"""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_analyze_query_node,
    make_build_context_node,
    make_extract_memory_node,
    make_filter_temporal_node,
    make_generate_response_node,
    make_ingest_event_node,
    make_judge_and_apply_node,
    make_retrieve_memory_node,
    make_traverse_graph_node,
)
from app.agent.state import IngestState, QueryState
from app.judgment.interface import BaseMemoryJudge
from app.llm.interface import BaseLLMClient
from app.memory.repository import BaseMemoryRepository


def build_query_workflow(repository: BaseMemoryRepository, llm_client: BaseLLMClient):
    graph = StateGraph(QueryState)

    graph.add_node("ANALYZE_QUERY", make_analyze_query_node())
    graph.add_node("RETRIEVE_MEMORY", make_retrieve_memory_node(repository, llm_client))
    graph.add_node("TRAVERSE_GRAPH", make_traverse_graph_node(repository))
    graph.add_node("FILTER_TEMPORAL_MEMORY", make_filter_temporal_node())
    graph.add_node("BUILD_CONTEXT", make_build_context_node())
    graph.add_node("GENERATE_RESPONSE", make_generate_response_node(llm_client))

    graph.add_edge(START, "ANALYZE_QUERY")
    graph.add_edge("ANALYZE_QUERY", "RETRIEVE_MEMORY")
    graph.add_edge("RETRIEVE_MEMORY", "TRAVERSE_GRAPH")
    graph.add_edge("TRAVERSE_GRAPH", "FILTER_TEMPORAL_MEMORY")
    graph.add_edge("FILTER_TEMPORAL_MEMORY", "BUILD_CONTEXT")
    graph.add_edge("BUILD_CONTEXT", "GENERATE_RESPONSE")
    graph.add_edge("GENERATE_RESPONSE", END)

    return graph.compile()


def build_ingest_workflow(
    repository: BaseMemoryRepository,
    llm_client: BaseLLMClient,
    judge: BaseMemoryJudge,
    default_subject: str,
):
    graph = StateGraph(IngestState)

    graph.add_node("INGEST_EVENT", make_ingest_event_node(repository))
    graph.add_node("EXTRACT_MEMORY", make_extract_memory_node(llm_client, default_subject))
    graph.add_node("JUDGE_AND_APPLY", make_judge_and_apply_node(repository, llm_client, judge))

    graph.add_edge(START, "INGEST_EVENT")
    graph.add_edge("INGEST_EVENT", "EXTRACT_MEMORY")
    graph.add_edge("EXTRACT_MEMORY", "JUDGE_AND_APPLY")
    graph.add_edge("JUDGE_AND_APPLY", END)

    return graph.compile()
