"""Agent2 JEV Result Filtering (docs/IMPLEMENTATION.md §6.2).

Score-per-chunk post-tool filtering:
  1. Decomposes raw tool JSON into discrete chunks (e.g. one contact or company record per chunk).
  2. Scores each chunk concurrently via `jev_score` against the user query.
  3. Retains only chunks scoring above `threshold` (default 0.50), sorted descending, capped at top-N (default 5).
  4. Reassembles surviving chunks into a compact block for LLM context.
  5. Computes token savings: `raw_result_tokens` vs `filtered_result_tokens`.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.cost import estimate_tokens
from app.core.graph8_client import ToolCallResult
from app.core.jev_client import JEVClient
from app.tracing.langsmith import traced

SCORE_THRESHOLD = 0.50
MAX_CHUNKS_KEPT = 5
MAX_CHUNKS_EVALUATED = 8  # cap Jev volume per turn to bound latency/costs


@dataclass
class FilterResult:
    """Outcome of filtering raw tool payload into compact context."""

    filtered_text: str
    raw_tokens: int
    filtered_tokens: int
    chunks_total: int = 0
    chunks_kept: int = 0
    latency_ms: float = 0.0
    used_fallback: bool = False
    notes: List[str] = field(default_factory=list)


def _decompose_into_chunks(raw_text: str) -> List[Dict[str, Any]]:
    """Parse JSON and extract discrete entity chunks (records/items)."""
    text = (raw_text or "").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except Exception:
        # Plain text fallback: split non-empty lines as chunk objects
        return [{"raw_line": line.strip()} for line in text.splitlines() if line.strip()][:MAX_CHUNKS_EVALUATED]

    if isinstance(data, list):
        return [item if isinstance(item, dict) else {"item": item} for item in data[:MAX_CHUNKS_EVALUATED]]

    if isinstance(data, dict):
        # Check for list properties commonly returned by Graph8 or CRM tools
        for key in ("companies", "contacts", "data", "results", "items"):
            val = data.get(key)
            if isinstance(val, list) and val:
                return [item if isinstance(item, dict) else {"item": item} for item in val[:MAX_CHUNKS_EVALUATED]]

        # Single record: break into key-value field pairs
        chunks = []
        for k, v in data.items():
            if v is not None and v != "" and v != []:
                chunks.append({k: v})
        return chunks[:MAX_CHUNKS_EVALUATED]

    return [{"data": data}]


@traced(name="agent2.jev_filter_result")
async def filter_tool_result(
    jev: JEVClient,
    tool_name: str,
    raw_result: ToolCallResult,
    user_message: str,
    run_group_id: str = "",
    threshold: float = SCORE_THRESHOLD,
    max_kept: int = MAX_CHUNKS_KEPT,
) -> FilterResult:
    """Run concurrent JEV Score over tool result chunks (§6.2)."""
    raw_text = raw_result.raw_text or ""
    raw_tokens = estimate_tokens(raw_text)

    if not raw_text or not raw_result.ok:
        return FilterResult(
            filtered_text=raw_text or f"Tool error: {raw_result.error}",
            raw_tokens=raw_tokens,
            filtered_tokens=raw_tokens,
            chunks_total=0,
            chunks_kept=0,
            notes=["empty_or_failed_tool_result"],
        )

    chunks = _decompose_into_chunks(raw_text)
    if not chunks:
        return FilterResult(
            filtered_text=raw_text,
            raw_tokens=raw_tokens,
            filtered_tokens=raw_tokens,
            chunks_total=0,
            chunks_kept=0,
            notes=["no_chunks_decomposed"],
        )

    question = f"Is this CRM data relevant to answering the user query: '{user_message}'?"
    start = time.perf_counter()

    async def _score_chunk(chunk: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
        res = await jev.jev_score(
            question=question,
            context={"tool": tool_name, "chunk": chunk},
            call_site="result_filtering",
            agent_id="agent2",
            run_group_id=run_group_id,
        )
        if not res.ok:
            return chunk, 0.0, False
        return chunk, res.normalized, True

    results = await asyncio.gather(*[_score_chunk(chunk) for chunk in chunks])
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    successful_scores = [r for r in results if r[2]]

    # If all JEV score calls failed (e.g. gateway error/403), gracefully fall back
    if not successful_scores:
        fallback_chunks = chunks[:3]
        fallback_text = json.dumps(fallback_chunks, indent=2)
        filtered_tokens = estimate_tokens(fallback_text)
        return FilterResult(
            filtered_text=fallback_text,
            raw_tokens=raw_tokens,
            filtered_tokens=filtered_tokens,
            chunks_total=len(chunks),
            chunks_kept=len(fallback_chunks),
            latency_ms=latency_ms,
            used_fallback=True,
            notes=["jev_scoring_unavailable_truncated_chunks_fallback"],
        )

    scored_chunks = [(c, s) for c, s, ok in results if ok and s >= threshold]
    scored_chunks.sort(key=lambda item: item[1], reverse=True)
    kept_chunks = [c for c, _ in scored_chunks[:max_kept]]

    if not kept_chunks:
        scored_all = [(c, s) for c, s, ok in results if ok]
        scored_all.sort(key=lambda item: item[1], reverse=True)
        kept_chunks = [scored_all[0][0]] if scored_all else chunks[:1]

    filtered_text = json.dumps(kept_chunks, indent=2)
    filtered_tokens = estimate_tokens(filtered_text)

    return FilterResult(
        filtered_text=filtered_text,
        raw_tokens=raw_tokens,
        filtered_tokens=filtered_tokens,
        chunks_total=len(chunks),
        chunks_kept=len(kept_chunks),
        latency_ms=latency_ms,
        used_fallback=False,
        notes=[f"jev_filtered: kept {len(kept_chunks)}/{len(chunks)} chunks, tokens {raw_tokens}->{filtered_tokens}"],
    )
