"""Typed JEV client for the demo (docs/IMPLEMENTATION.md §4.2).

Wraps the real Vercel AI Gateway evaluation endpoint
(``POST {JEV_URL}`` for ``typesafe-ai/jev``) and exposes Jev's three
primitives as typed methods:

  * ``jev_choice(question, options, context)`` -> selected option + calibrated
    probability per option
  * ``jev_score(question, context)``           -> 0-1 calibrated score
  * ``jev_noul(question, context)``            -> boolean + confidence

Wire names: the gateway accepts question types ``choice`` | ``score`` |
``boolean``. "Noul" is TypeSafe's name for its boolean primitive, and the
gateway answers it with a single calibrated yes-probability — so `jev_noul`
speaks ``type: "boolean"`` on the wire and tolerates both ``boolean``/``noul``
answer field spellings.

Every call is logged with what the dashboard needs: estimated input tokens
(Jev bills per input token, $0 output), latency, the call site it came from
(``tool_routing`` | ``result_filtering`` | ``staleness_check``), and the real
error text when the endpoint refuses (e.g. a free-tier 403 for `typesafe-ai/jev`
— which this build surfaces rather than hides, so the dashboard can show
"JEV unavailable" instead of inventing judgments).

Calls that fail return a result object with ``ok=False`` and an ``error``; the
callers (§6.2-6.4) decide how to proceed, and the metrics collector counts the
failure. Nothing here is mocked.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from app.core.cost import estimate_tokens
from app.core.tracing import child_trace

JEV_MODEL = "jev"
CALL_SITES = ("tool_routing", "result_filtering", "staleness_check")

# Default rubric for `jev_score` when the caller doesn't supply one. Three
# levels is enough granularity for chunk relevance and keeps Jev's latency low.
DEFAULT_SCORE_LEVELS = ["not relevant", "partially relevant", "directly relevant"]


def _choice_question(instructions: str, criteria: Dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _score_question(instructions: str, levels: List[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def _boolean_question(instructions: str, true_meaning: str, false_meaning: str) -> dict:
    return {
        "type": "boolean",
        "instructions": instructions,
        "criteria": {"true": true_meaning, "false": false_meaning},
    }


@dataclass
class ChoiceResult:
    """A pick from a fixed label set, with Jev's calibrated probabilities."""

    ok: bool
    choice: str = ""
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens_estimate: int = 0
    call_site: str = ""
    error: Optional[str] = None
    raw_answer: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """A rubric score, plus its normalized 0-1 form.

    Jev returns an *expected* score (a probability-weighted average of the
    rubric levels, e.g. 1.7 on a 0-2 rubric) with the probabilities it was
    computed from. `normalized` is that score mapped onto 0..1 by the number of
    rubric levels — kept alongside the raw values so the dashboard can show the
    math rather than just the conclusion.
    """

    ok: bool
    score: float = 0.0
    normalized: float = 0.0
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    levels: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens_estimate: int = 0
    call_site: str = ""
    error: Optional[str] = None
    raw_answer: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NoulResult:
    """Yes/no with the calibrated probability of "yes" behind it.

    `confidence` is the decisiveness of that probability (distance from the 0.5
    decision boundary), documented here because it is derived, not returned.
    """

    ok: bool
    value: bool = False
    probability: float = 0.5
    confidence: float = 0.0
    latency_ms: float = 0.0
    input_tokens_estimate: int = 0
    call_site: str = ""
    error: Optional[str] = None
    raw_answer: Dict[str, Any] = field(default_factory=dict)


class JEVClient:
    """One JEV client, shared by both agents (Agent1 never calls it; Agent2
    uses it for tool routing, result filtering, and staleness checks)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: str = "https://ai-gateway.vercel.sh/v1/evaluate",
        model: str = JEV_MODEL,
        timeout: float = 20.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._timeout = timeout
        self._transport = transport
        self._calls_by_site: Dict[str, int] = {site: 0 for site in CALL_SITES}
        self._errors_by_site: Dict[str, int] = {site: 0 for site in CALL_SITES}
        self._tokens_estimate_total = 0
        self._latency_ms_total = 0.0
        self._last_latency_ms = 0.0
        self._last_tokens_estimate = 0
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    # -- raw call ---------------------------------------------------------------

    async def evaluate(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        call_site: str,
        agent_id: str = "agent2",
        run_group_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """POST one System One request. Returns None on any failure (never raises)."""
        if not self._api_key:
            self.last_error = "no JEV API key configured"
            return None
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json={"model": self._model, "state": state, "questions": questions},
                )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self._latency_ms_total += (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            self.last_error = f"http_{response.status_code}: {response.text[:300]}"
            return None
        try:
            return response.json()
        except ValueError:
            self.last_error = f"unparseable response: {response.text[:200]}"
            return None

    # -- primitives -------------------------------------------------------------

    async def jev_choice(
        self,
        question: str,
        options: List[str],
        context: Dict[str, Any],
        call_site: str,
        option_criteria: Optional[Dict[str, str]] = None,
        instructions: Optional[str] = None,
        agent_id: str = "agent2",
        run_group_id: str = "",
    ) -> ChoiceResult:
        """Pick one of `options` given `context`."""
        criteria = {option: (option_criteria or {}).get(option, option) for option in options}
        payload = await self._timed_call(
            question=question,
            question_payload=_choice_question(instructions or question, criteria),
            state=context,
            call_site=call_site,
            agent_id=agent_id,
            run_group_id=run_group_id,
        )
        if payload is None:
            return ChoiceResult(ok=False, call_site=call_site, error=self.last_error)
        answer = self._answer(payload)
        if not answer:
            return ChoiceResult(ok=False, call_site=call_site, error="empty answers", raw_answer=payload)
        choice = str(answer.get("choice") or "")
        if choice not in criteria:
            return ChoiceResult(
                ok=False,
                call_site=call_site,
                error=f"choice {choice!r} not in options",
                raw_answer=answer,
                choice=choice,
            )
        probabilities = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        return ChoiceResult(
            ok=True,
            choice=choice,
            confidence=float(answer.get("confidence") or probabilities.get(choice) or 0.0),
            probabilities=probabilities,
            latency_ms=self._last_latency_ms,
            input_tokens_estimate=self._last_tokens_estimate,
            call_site=call_site,
            raw_answer=answer,
        )

    async def jev_score(
        self,
        question: str,
        context: Dict[str, Any],
        call_site: str,
        levels: Optional[List[str]] = None,
        agent_id: str = "agent2",
        run_group_id: str = "",
    ) -> ScoreResult:
        """Score `context` against an ordered rubric, and normalize to 0..1."""
        rubric = list(levels or DEFAULT_SCORE_LEVELS)
        payload = await self._timed_call(
            question=question,
            question_payload=_score_question(question, rubric),
            state=context,
            call_site=call_site,
            agent_id=agent_id,
            run_group_id=run_group_id,
        )
        if payload is None:
            return ScoreResult(ok=False, call_site=call_site, error=self.last_error, levels=rubric)
        answer = self._answer(payload)
        if not answer or answer.get("score") is None:
            return ScoreResult(ok=False, call_site=call_site, error="no score in answer", raw_answer=payload, levels=rubric)
        raw_score = float(answer.get("score") or 0.0)
        top = max(1, len(rubric) - 1)
        return ScoreResult(
            ok=True,
            score=raw_score,
            normalized=max(0.0, min(1.0, raw_score / top)),
            confidence=float(answer.get("confidence") or 0.0),
            probabilities={str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()},
            levels=rubric,
            latency_ms=self._last_latency_ms,
            input_tokens_estimate=self._last_tokens_estimate,
            call_site=call_site,
            raw_answer=answer,
        )

    async def jev_noul(
        self,
        question: str,
        context: Dict[str, Any],
        call_site: str,
        true_meaning: str = "yes / the statement is true",
        false_meaning: str = "no / the statement is false",
        agent_id: str = "agent2",
        run_group_id: str = "",
    ) -> NoulResult:
        """Boolean judgment with Jev's calibrated probability of "yes"."""
        payload = await self._timed_call(
            question=question,
            question_payload=_boolean_question(question, true_meaning, false_meaning),
            state=context,
            call_site=call_site,
            agent_id=agent_id,
            run_group_id=run_group_id,
        )
        if payload is None:
            return NoulResult(ok=False, call_site=call_site, error=self.last_error)
        answer = self._answer(payload)
        if not answer:
            return NoulResult(ok=False, call_site=call_site, error="empty answers", raw_answer=payload)
        probability = self._boolean_probability(answer)
        if probability is None:
            return NoulResult(
                ok=False,
                call_site=call_site,
                error="no boolean probability in answer",
                raw_answer=answer,
            )
        return NoulResult(
            ok=True,
            value=probability >= 0.5,
            probability=probability,
            confidence=abs(probability - 0.5) * 2.0,  # decisiveness, documented in the dataclass
            latency_ms=self._last_latency_ms,
            input_tokens_estimate=self._last_tokens_estimate,
            call_site=call_site,
            raw_answer=answer,
        )

    # -- internals --------------------------------------------------------------

    async def _timed_call(
        self,
        question: str,
        question_payload: Dict[str, Any],
        state: Dict[str, Any],
        call_site: str,
        agent_id: str,
        run_group_id: str,
    ) -> Optional[Dict[str, Any]]:
        """One traced, accounted JEV call (docs §4.2: token estimate, latency,
        call site, LangSmith span tagged component="jev")."""
        self._calls_by_site[call_site] = self._calls_by_site.get(call_site, 0) + 1
        estimate = estimate_tokens(json.dumps({"state": state, "questions": question_payload}, default=str))
        started = time.perf_counter()
        with child_trace(
            name=f"jev.{call_site}",
            run_type="chain",
            agent_id=agent_id,
            run_group_id=run_group_id,
            component="jev",
            call_type=call_site,
            metadata={"question": question, "input_tokens_estimate": estimate, "model": self._model},
        ) as run:
            payload = await self.evaluate(state, {"answer": question_payload}, call_site, agent_id, run_group_id)
            self._last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._last_tokens_estimate = estimate
            if payload is None:
                self._errors_by_site[call_site] = self._errors_by_site.get(call_site, 0) + 1
            else:
                usage = payload.get("usage") or {}
                reported = usage.get("input_tokens")
                if isinstance(reported, (int, float)) and reported > 0:
                    self._last_tokens_estimate = int(reported)
                self._tokens_estimate_total += self._last_tokens_estimate
            try:
                run.outputs = {
                    "ok": payload is not None,
                    "answer": (payload or {}).get("answers", {}).get("answer"),
                    "error": self.last_error if payload is None else None,
                    "latency_ms": self._last_latency_ms,
                }
            except Exception:
                pass  # tracing must never break a real JEV call
        return payload

    @staticmethod
    def _answer(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        answers = payload.get("answers") or {}
        answer = answers.get("answer") or (next(iter(answers.values())) if answers else None)
        return answer if isinstance(answer, dict) else {}

    @staticmethod
    def _boolean_probability(answer: Dict[str, Any]) -> Optional[float]:
        """Jev's boolean primitive: a calibration float under one of the field
        names in use (`boolean` on the gateway, `noul` in TypeSafe's own SDK)."""
        for key in ("boolean", "noul", "value", "probability"):
            value = answer.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        return None

    def stats(self) -> Dict[str, Any]:
        """Per-call-site accounting the metrics dashboard renders (docs §4.2)."""
        return {
            "configured": self.configured,
            "model": self._model,
            "url": self._url,
            "calls_by_site": dict(self._calls_by_site),
            "errors_by_site": dict(self._errors_by_site),
            "calls_total": sum(self._calls_by_site.values()),
            "errors_total": sum(self._errors_by_site.values()),
            "input_tokens_estimate_total": self._tokens_estimate_total,
            "input_cost_usd_note": "Jev bills per input token, $0 output tokens — token counts here are the real usage when the gateway reports it, else a ~4 chars/token estimate.",
            "latency_ms_total": round(self._latency_ms_total, 2),
            "last_error": self.last_error,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "calls_by_site": dict(self._calls_by_site),
            "errors_by_site": dict(self._errors_by_site),
            "total_calls": sum(self._calls_by_site.values()),
            "total_errors": sum(self._errors_by_site.values()),
            "input_tokens_estimate_total": self._tokens_estimate_total,
            "latency_ms_total": round(self._latency_ms_total, 2),
        }

    def delta(self, before: Dict[str, Any]) -> Dict[str, Any]:
        """What Jev calls cost for one turn: `after.delta(before)`."""
        prev_calls = before.get("calls_by_site") or {}
        prev_errors = before.get("errors_by_site") or {}
        by_site = {site: self._calls_by_site.get(site, 0) - prev_calls.get(site, 0) for site in CALL_SITES}
        errors_by_site = {site: self._errors_by_site.get(site, 0) - prev_errors.get(site, 0) for site in CALL_SITES}
        return {
            "by_site": by_site,
            "errors_by_site": errors_by_site,
            "total_calls": sum(by_site.values()),
            "total_errors": sum(errors_by_site.values()),
            "tokens_estimate": self._tokens_estimate_total - int(before.get("input_tokens_estimate_total", 0)),
            "latency_ms": round(self._latency_ms_total - float(before.get("latency_ms_total", 0.0)), 2),
        }



