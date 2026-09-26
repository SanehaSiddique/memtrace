"""Builds the minimal, explainable context handed to the LLM.

The agent never sees the whole memory store — only `selected`, already
temporally filtered and capped to `max_memories`. `excluded` is kept
alongside for the debug/explain surface ("why wasn't MongoDB included?").
"""

from typing import List, Optional

from app.memory.models import ContextResult, ExcludedMemory, ScoredMemory

DEFAULT_MAX_MEMORIES = 6


def estimate_tokens(text: str) -> int:
    # ~4 chars/token is a reasonable heuristic without pulling in a tokenizer dependency.
    return max(1, len(text) // 4) if text else 0


def build_context(
    query: str,
    selected: List[ScoredMemory],
    excluded: List[ExcludedMemory],
    recent_messages: Optional[List[str]] = None,
    max_memories: int = DEFAULT_MAX_MEMORIES,
) -> ContextResult:
    kept = selected[:max_memories]
    overflowed = selected[max_memories:]

    all_excluded = list(excluded) + [
        ExcludedMemory(memory=sm.memory, reason=f"Below context budget (rank > {max_memories}).")
        for sm in overflowed
    ]

    lines = [f"USER QUERY: {query}", "", "RELEVANT MEMORY:"]
    if not kept:
        lines.append("(no memory met the relevance/validity bar for this query)")
    for sm in kept:
        m = sm.memory
        lines.append(
            f"- [{m.status.value}] {m.subject} {m.predicate} {m.object} "
            f"— {m.content} (confidence={m.confidence:.2f}, score={sm.score:.2f}, "
            f"reasons={','.join(sm.retrieval_reason)})"
        )

    if recent_messages:
        lines.append("")
        lines.append("RECENT CONVERSATION:")
        lines.extend(f"- {msg}" for msg in recent_messages[-5:])

    context_text = "\n".join(lines)

    return ContextResult(
        context_text=context_text,
        selected=kept,
        excluded=all_excluded,
        token_estimate=estimate_tokens(context_text),
    )
