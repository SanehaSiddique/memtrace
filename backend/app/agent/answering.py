"""Deterministic, offline-safe answer synthesis from selected memories.

Used by the GENERATE_RESPONSE node when no live LLM key is configured, and by
the evaluation runner so baseline-vs-MEMTRACE comparisons don't depend on a
live model either.
"""

from typing import List

from app.memory.models import MemoryStatus, ScoredMemory


def synthesize_offline_answer(selected: List[ScoredMemory]) -> str:
    if not selected:
        return "I don't have a currently valid memory to answer that."

    active = [s for s in selected if s.memory.status == MemoryStatus.ACTIVE]
    historical = [s for s in selected if s.memory.status == MemoryStatus.HISTORICAL]
    top = active[0] if active else selected[0]

    sentence = f"{top.memory.subject} {top.memory.predicate.replace('_', ' ')} {top.memory.object}."

    # extraction_reason is the "because X" clause pulled from the source sentence itself;
    # judgment_reason is the judge's internal rationale for the lifecycle operation and
    # isn't meant to be read as a user-facing explanation, so it's deliberately excluded here.
    reason = top.memory.provenance.get("extraction_reason")
    if reason:
        sentence += f" This was because {reason}."

    if historical and historical[0].memory.id != top.memory.id:
        sentence += f" Previously it was {historical[0].memory.object}, since superseded."

    rejected = next((s for s in selected if s.memory.predicate == "rejected" and s.memory.id != top.memory.id), None)
    if rejected:
        sentence += f" {rejected.memory.object} was considered and rejected."

    return sentence
