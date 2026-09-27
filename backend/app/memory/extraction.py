"""Turn a raw conversation/event into candidate structured memories.

Deterministic, rule-based extraction tuned to the kind of statements MEMTRACE's
demo story contains (technology/decision changes: "we moved from X to Y",
"we chose X", "X depends on Y", ...). This keeps the pipeline fully runnable
without an LLM key. When a live LLM client is available, extraction is
attempted there first and falls back to the rules below if the model's output
can't be parsed as JSON.
"""

import json
import re
from typing import List, Optional

from app.llm.interface import BaseLLMClient
from app.memory.models import CandidateMemory, MemoryEvent, MemoryType

_DOMAIN_PREDICATES = [
    ({"mongodb", "postgresql", "postgres", "mysql", "sqlite", "dynamodb"}, "uses_database"),
    ({"aws", "railway", "heroku", "vercel", "gcp", "azure", "digitalocean"}, "deployed_on"),
    ({"jwt", "oauth", "session-based auth", "session auth", "session"}, "uses_auth"),
]

_MIGRATION_RE = re.compile(
    r"(?:moved|migrat\w+|switch\w+|chang\w+)\s+(?:from\s+)?(?P<old>[\w .+-]+?)\s+to\s+(?P<new>[\w .+-]+?)"
    r"(?:\s+because\s+(?P<reason>.+?))?[\.\!]?\s*$",
    re.IGNORECASE,
)
_SELECTED_RE = re.compile(
    r"(?:chose|selected|decided (?:to use|on))\s+(?P<new>[\w .+-]+?)"
    r"(?:\s+over\s+(?P<old>[\w .+-]+?))?(?:\s+because\s+(?P<reason>.+?))?[\.\!]?\s*$",
    re.IGNORECASE,
)
_CONSIDERING_RE = re.compile(
    r"(?:consider\w*|evaluat\w*|explor\w*)\s+(?P<candidate>[\w .+-]+?)(?:\s+because\s+(?P<reason>.+?))?[\.\!]?\s*$",
    re.IGNORECASE,
)
_USES_RE = re.compile(
    r"(?P<subject>[A-Z][\w ]*?)\s+(?P<verb>uses|depends on|runs on)\s+(?P<object>[\w .+-]+?)[\.\!]?\s*$"
)
_VERB_DEFAULT_PREDICATE = {"uses": "uses", "depends on": "depends_on", "runs on": "runs_on"}
_REJECTED_RE = re.compile(r"reject\w*\s+(?P<object>[\w .+-]+?)(?:\s+because\s+(?P<reason>.+?))?[\.\!]?\s*$", re.IGNORECASE)


def _domain_predicate(*phrases: str) -> str:
    haystack = " ".join(p.lower() for p in phrases if p)
    for keywords, predicate in _DOMAIN_PREDICATES:
        if any(k in haystack for k in keywords):
            return predicate
    return "uses"


def _extract_rule_based(content: str, default_subject: str) -> List[CandidateMemory]:
    text = content.strip()
    candidates: List[CandidateMemory] = []

    m = _MIGRATION_RE.search(text)
    if m:
        old, new, reason = m.group("old").strip(), m.group("new").strip(), m.group("reason")
        predicate = _domain_predicate(old, new)
        candidates.append(
            CandidateMemory(
                memory_type=MemoryType.FACT,
                subject=default_subject,
                predicate=predicate,
                object=new,
                content=text,
                confidence=0.9,
                extraction_reason=(reason or "").strip(),
            )
        )
        return candidates

    m = _SELECTED_RE.search(text)
    if m:
        new, reason = m.group("new").strip(), m.group("reason")
        predicate = _domain_predicate(new)
        candidates.append(
            CandidateMemory(
                memory_type=MemoryType.DECISION,
                subject=default_subject,
                predicate=predicate if predicate != "uses" else "selected",
                object=new,
                content=text,
                confidence=0.85,
                extraction_reason=(reason or "").strip(),
            )
        )
        return candidates

    m = _REJECTED_RE.search(text)
    if m:
        obj, reason = m.group("object").strip(), m.group("reason")
        candidates.append(
            CandidateMemory(
                memory_type=MemoryType.DECISION,
                subject=default_subject,
                predicate="rejected",
                object=obj,
                content=text,
                confidence=0.8,
                extraction_reason=(reason or "").strip(),
            )
        )
        return candidates

    m = _CONSIDERING_RE.search(text)
    if m:
        candidate, reason = m.group("candidate").strip(), m.group("reason")
        candidates.append(
            CandidateMemory(
                memory_type=MemoryType.GOAL,
                subject=default_subject,
                predicate="considering",
                object=candidate,
                content=text,
                confidence=0.6,
                extraction_reason=(reason or "").strip(),
            )
        )
        return candidates

    m = _USES_RE.search(text)
    if m:
        verb = m.group("verb").lower()
        default_predicate = _VERB_DEFAULT_PREDICATE[verb]
        domain_match = _domain_predicate(m.group("object"))
        predicate = domain_match if (verb == "uses" and domain_match != "uses") else default_predicate
        candidates.append(
            CandidateMemory(
                memory_type=MemoryType.FACT,
                subject=m.group("subject").strip(),
                predicate=predicate,
                object=m.group("object").strip(),
                content=text,
                confidence=0.75,
            )
        )
        return candidates

    # Fallback: low-confidence generic fact so nothing is silently dropped;
    # the judge will likely route this to PENDING_REVIEW rather than ACTIVE.
    candidates.append(
        CandidateMemory(
            memory_type=MemoryType.EVENT_SUMMARY,
            subject=default_subject,
            predicate="mentioned",
            object=text[:120],
            content=text,
            confidence=0.4,
        )
    )
    return candidates


_LLM_SYSTEM_PROMPT = (
    "Extract structured (subject, predicate, object) memory candidates from the message. "
    'Respond ONLY with a JSON array of objects: '
    '[{"memory_type": "FACT|DECISION|PREFERENCE|GOAL|CONSTRAINT|EVENT_SUMMARY", '
    '"subject": str, "predicate": str, "object": str, "content": str, "confidence": float}]'
)

async def extract_candidates(
    event: MemoryEvent,
    default_subject: str,
    llm_client: Optional[BaseLLMClient] = None,
) -> List[CandidateMemory]:
    if llm_client is not None and llm_client.is_live:
        try:
            raw = await llm_client.chat(_LLM_SYSTEM_PROMPT, event.content)
            payload = json.loads(_strip_code_fence(raw))
            return [CandidateMemory(**item) for item in payload]
        except Exception:
            pass  # model unavailable/rate-limited/unparseable → deterministic rules
    return _extract_rule_based(event.content, default_subject)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.removeprefix("json").strip() if text.lower().startswith("json") else text
    return text
