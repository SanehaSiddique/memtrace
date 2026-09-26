"""Runs EVAL_CASES against a seeded repository, comparing:

  BASELINE  — pure semantic top-k (retrieval.baseline_semantic_retrieve), no
              graph traversal, no status/temporal awareness at all.
  MEMTRACE  — full hybrid pipeline: semantic + graph traversal + temporal
              filtering, exactly what app.agent.workflow's query graph runs.

Both sides answer with the same deterministic offline synthesizer so the
comparison isolates retrieval quality, not LLM phrasing. Real measured
numbers only — nothing here is a fabricated "improvement".
"""

import time
from typing import List

from pydantic import BaseModel

from app.agent.answering import synthesize_offline_answer
from app.evaluation.dataset import EVAL_CASES, EvalCase
from app.llm.interface import BaseLLMClient
from app.memory.repository import BaseMemoryRepository
from app.memory.retrieval import (
    apply_temporal_filter,
    baseline_semantic_retrieve,
    retrieve_semantic,
    score_and_rank,
    traverse_graph,
)


class CaseResult(BaseModel):
    name: str
    category: str
    query: str
    baseline_answer: str
    baseline_correct: bool
    baseline_latency_ms: float
    memtrace_answer: str
    memtrace_correct: bool
    memtrace_latency_ms: float
    memtrace_token_estimate: int


class EvaluationReport(BaseModel):
    cases: List[CaseResult]
    baseline_accuracy: float
    memtrace_accuracy: float
    avg_memtrace_token_estimate: float
    avg_baseline_latency_ms: float
    avg_memtrace_latency_ms: float


def _judge(answer: str, case: EvalCase) -> bool:
    lowered = answer.lower()
    if not all(term.lower() in lowered for term in case.expect_contains):
        return False
    if any(term.lower() in lowered for term in case.expect_excludes):
        return False
    return True


async def _run_memtrace(repository, llm_client, case, agent_id: str):
    start = time.perf_counter()
    candidates = await retrieve_semantic(repository, llm_client, case.query, agent_id)
    candidates = await traverse_graph(repository, candidates)
    ranked = score_and_rank(candidates, top_k=8)
    selected, _ = apply_temporal_filter(case.query, ranked)
    answer = synthesize_offline_answer(selected)
    latency_ms = (time.perf_counter() - start) * 1000
    token_estimate = max(1, len(answer) // 4)
    return answer, latency_ms, token_estimate


async def _run_baseline(repository, llm_client, case, agent_id: str):
    start = time.perf_counter()
    hits = await baseline_semantic_retrieve(repository, llm_client, case.query, agent_id, top_k=3)
    answer = synthesize_offline_answer(hits)
    latency_ms = (time.perf_counter() - start) * 1000
    return answer, latency_ms


async def run_evaluation(
    repository: BaseMemoryRepository,
    llm_client: BaseLLMClient,
    agent_id: str,
    cases: List[EvalCase] = EVAL_CASES,
) -> EvaluationReport:
    results: List[CaseResult] = []
    for case in cases:
        baseline_answer, baseline_latency = await _run_baseline(repository, llm_client, case, agent_id)
        memtrace_answer, memtrace_latency, token_estimate = await _run_memtrace(repository, llm_client, case, agent_id)

        results.append(
            CaseResult(
                name=case.name,
                category=case.category,
                query=case.query,
                baseline_answer=baseline_answer,
                baseline_correct=_judge(baseline_answer, case),
                baseline_latency_ms=baseline_latency,
                memtrace_answer=memtrace_answer,
                memtrace_correct=_judge(memtrace_answer, case),
                memtrace_latency_ms=memtrace_latency,
                memtrace_token_estimate=token_estimate,
            )
        )

    n = len(results) or 1
    return EvaluationReport(
        cases=results,
        baseline_accuracy=sum(r.baseline_correct for r in results) / n,
        memtrace_accuracy=sum(r.memtrace_correct for r in results) / n,
        avg_memtrace_token_estimate=sum(r.memtrace_token_estimate for r in results) / n,
        avg_baseline_latency_ms=sum(r.baseline_latency_ms for r in results) / n,
        avg_memtrace_latency_ms=sum(r.memtrace_latency_ms for r in results) / n,
    )
