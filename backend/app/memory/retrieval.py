"""Hybrid, explainable memory retrieval.

Combines three independent signals instead of a single vector-similarity
top-k:

  semantic_score  — cosine similarity against stored (or hash-based mock)
                     embeddings, plus a keyword-overlap fallback so retrieval
                     stays useful even with the offline mock embedder.
  graph_score     — memories reachable from a semantic hit via one relationship
                     hop (e.g. the PostgreSQL memory pulls in the MongoDB
                     memory it SUPERSEDES, and vice versa).
  status_score    — ACTIVE is boosted, HISTORICAL/ARCHIVED/PENDING_REVIEW are
                     penalized, so temporal validity participates in ranking
                     even before the explicit temporal filter runs.

Every ScoredMemory keeps `retrieval_reason` so the debug endpoint can say
exactly why a memory surfaced.
"""

import re
from typing import List, Tuple

from app.llm.interface import BaseLLMClient
from app.memory.models import ExcludedMemory, Memory, MemoryStatus, ScoredMemory
from app.memory.repository import BaseMemoryRepository

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HISTORY_INTENT_WORDS = {
    "before", "previously", "used", "originally", "history", "historical",
    "past", "old", "former", "formerly", "prior",
}
_STOPWORDS = {
    "what", "are", "we", "is", "the", "a", "an", "of", "to", "for", "in", "on",
    "now", "did", "do", "does", "currently", "still", "this", "that", "it",
    "and", "or", "with", "our", "us", "was", "were", "have", "has", "had",
    "about", "any", "team",
}
_STEM_LEN = 4


def _keyword_stems(text: str) -> set:
    """Crude prefix stemmer + stopword filter so "deployment"/"deployed" or
    "auth"/"authentication" overlap without needing a real NLP pipeline."""
    return {
        token[:_STEM_LEN]
        for token in _tokenize(text)
        if token not in _STOPWORDS and len(token) > 2
    }

STATUS_SCORES = {
    MemoryStatus.ACTIVE: 0.3,
    MemoryStatus.HISTORICAL: -0.2,
    MemoryStatus.PENDING_REVIEW: -0.4,
    MemoryStatus.ARCHIVED: -0.6,
    MemoryStatus.DELETED: -1.0,
}
STATUS_REASON = {
    MemoryStatus.ACTIVE: "active_memory",
    MemoryStatus.HISTORICAL: "historical_memory",
    MemoryStatus.PENDING_REVIEW: "pending_review",
    MemoryStatus.ARCHIVED: "archived_memory",
    MemoryStatus.DELETED: "deleted_memory",
}


def _tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def wants_historical_context(query: str) -> bool:
    return bool(_tokenize(query) & _HISTORY_INTENT_WORDS)


async def retrieve_semantic(
    repository: BaseMemoryRepository,
    llm_client: BaseLLMClient,
    query: str,
    agent_id: str,
    keyword_scan_limit: int = 500,
) -> dict[str, ScoredMemory]:
    """Signal 1 & 2: semantic embedding similarity, backstopped by keyword overlap."""
    candidates: dict[str, ScoredMemory] = {}

    query_embedding = await llm_client.embed(query)
    for memory, sem_score in await repository.search_semantic(agent_id, query_embedding, limit=20):
        if sem_score <= 0.2:
            continue
        candidates[memory.id] = ScoredMemory(
            memory=memory,
            score=0.0,
            retrieval_reason=["semantic_match"],
            semantic_score=sem_score,
        )

    query_tokens = _keyword_stems(query)
    if query_tokens:
        for memory in await repository.list_memories(agent_id, limit=keyword_scan_limit):
            memory_tokens = _keyword_stems(f"{memory.subject} {memory.predicate} {memory.object} {memory.content}")
            overlap = query_tokens & memory_tokens
            if not overlap:
                continue
            keyword_score = min(1.0, 0.3 + len(overlap) / max(len(memory_tokens), 1))
            existing = candidates.get(memory.id)
            if existing is None:
                candidates[memory.id] = ScoredMemory(
                    memory=memory,
                    score=0.0,
                    retrieval_reason=["keyword_match"],
                    semantic_score=keyword_score,
                )
            else:
                existing.semantic_score = max(existing.semantic_score, keyword_score)
                if "keyword_match" not in existing.retrieval_reason:
                    existing.retrieval_reason.append("keyword_match")

    return candidates


async def traverse_graph(
    repository: BaseMemoryRepository,
    candidates: dict[str, ScoredMemory],
) -> dict[str, ScoredMemory]:
    """Signal 3: pull in one-hop graph neighbors of whatever semantic/keyword retrieval found
    (e.g. the ACTIVE PostgreSQL memory pulls in the HISTORICAL MongoDB memory it SUPERSEDES)."""
    expanded = dict(candidates)
    for memory_id in list(candidates.keys()):
        for rel, other in await repository.get_related_memories(memory_id, direction="both"):
            if other is None or other.id in expanded:
                continue
            expanded[other.id] = ScoredMemory(
                memory=other,
                score=0.0,
                retrieval_reason=["graph_neighbor"],
                graph_score=0.4,
                graph_distance=1,
            )
    return expanded


def score_and_rank(candidates: dict[str, ScoredMemory], top_k: int = 8) -> List[ScoredMemory]:
    """Signal 4: temporal/status validity, folded into the final ranking score."""
    for scored in candidates.values():
        status = scored.memory.status
        scored.status_score = STATUS_SCORES.get(status, 0.0)
        reason = STATUS_REASON.get(status)
        if reason and reason not in scored.retrieval_reason:
            scored.retrieval_reason.append(reason)
        scored.score = scored.semantic_score + scored.graph_score + scored.status_score

    ranked = sorted(candidates.values(), key=lambda s: s.score, reverse=True)
    return ranked[:top_k]


async def baseline_semantic_retrieve(
    repository: BaseMemoryRepository,
    llm_client: BaseLLMClient,
    query: str,
    agent_id: str,
    top_k: int = 8,
) -> List[ScoredMemory]:
    """The "flat vector DB" baseline: pure embedding top-k, no graph, no status/temporal
    awareness at all. Used only by /debug/replay to make MEMTRACE's advantage measurable
    against something real, not a strawman."""
    query_embedding = await llm_client.embed(query)
    hits = await repository.search_semantic(agent_id, query_embedding, limit=top_k)
    return [
        ScoredMemory(memory=memory, score=sem_score, retrieval_reason=["semantic_match"], semantic_score=sem_score)
        for memory, sem_score in hits
    ]


async def hybrid_retrieve(
    repository: BaseMemoryRepository,
    llm_client: BaseLLMClient,
    query: str,
    agent_id: str,
    top_k: int = 8,
    keyword_scan_limit: int = 500,
) -> List[ScoredMemory]:
    """Convenience wrapper chaining retrieve_semantic -> traverse_graph -> score_and_rank.

    The LangGraph query workflow calls the three steps individually (as
    RETRIEVE_MEMORY / TRAVERSE_GRAPH nodes) so each is traced separately.
    """
    candidates = await retrieve_semantic(repository, llm_client, query, agent_id, keyword_scan_limit)
    candidates = await traverse_graph(repository, candidates)
    return score_and_rank(candidates, top_k)


def apply_temporal_filter(
    query: str,
    scored_memories: List[ScoredMemory],
) -> Tuple[List[ScoredMemory], List[ExcludedMemory]]:
    """Decide which retrieved memories are currently valid for this query.

    A HISTORICAL/ARCHIVED memory is excluded whenever an ACTIVE memory covers
    the same (subject, predicate) — unless the query itself asks about the
    past, in which case both are kept so the answer can contrast them.
    """
    wants_history = wants_historical_context(query)

    active_by_key: dict[Tuple[str, str], Memory] = {}
    for scored in scored_memories:
        if scored.memory.status == MemoryStatus.ACTIVE:
            key = (scored.memory.subject.lower(), scored.memory.predicate.lower())
            active_by_key[key] = scored.memory

    selected: List[ScoredMemory] = []
    excluded: List[ExcludedMemory] = []

    for scored in scored_memories:
        memory = scored.memory
        key = (memory.subject.lower(), memory.predicate.lower())

        if memory.status == MemoryStatus.DELETED:
            excluded.append(ExcludedMemory(memory=memory, reason="Memory was deleted."))
            continue

        if memory.status == MemoryStatus.PENDING_REVIEW:
            excluded.append(ExcludedMemory(memory=memory, reason="Memory is pending human review and not yet confirmed."))
            continue

        if memory.status in (MemoryStatus.HISTORICAL, MemoryStatus.ARCHIVED) and not wants_history:
            current = active_by_key.get(key)
            if current is not None and current.id != memory.id:
                excluded.append(
                    ExcludedMemory(
                        memory=memory,
                        reason=(
                            f"Superseded by '{current.object}' (memory {current.id}), which is now ACTIVE "
                            f"for {memory.subject}/{memory.predicate}; marked {memory.status.value}."
                        ),
                    )
                )
                continue
            if current is None:
                excluded.append(
                    ExcludedMemory(memory=memory, reason=f"Marked {memory.status.value}; no longer currently valid.")
                )
                continue

        selected.append(scored)

    return selected, excluded
