"""Agent1 memory: plain STM plus the naive Postgres fact ledger (docs §5.1).

STM is a plain chat-history list for the session — identical to Agent2's, so
the only deltas between the two agents are LTM structure and tool handling.

LTM is deliberately dumb:
  * `remember(facts)` appends rows; no update, no dedup, no staleness check,
    so contradictions accumulate forever;
  * `context_block()` renders *every* row for the session as plain text with no
    relevance filtering. This is the "seven tools, ten tools... all of it goes
    into the prompt every time" problem in its memory flavor, and it is on
    purpose: it is the flaw Agent2 exists to fix.
"""

from typing import Dict, List

from app.db.postgres import FactRecord, FactsStore


class Agent1Memory:
    """STM (in-process history) + LTM (flat facts table) for one session."""

    def __init__(self, facts_store: FactsStore) -> None:
        self._facts = facts_store
        self._stm: Dict[str, List[dict]] = {}  # session_id -> [{"role", "content"}]

    @property
    def facts_store(self) -> FactsStore:
        return self._facts

    # -- STM --------------------------------------------------------------------

    def history(self, session_id: str) -> List[dict]:
        return list(self._stm.get(session_id, []))

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        if user_message:
            self._stm.setdefault(session_id, []).append({"role": "user", "content": user_message})
        if assistant_message:
            self._stm.setdefault(session_id, []).append({"role": "assistant", "content": assistant_message})

    # -- LTM --------------------------------------------------------------------

    async def remember(self, session_id: str, facts: List[str]) -> List[FactRecord]:
        """Naive write: one new row per extracted fact. Never updates, never dedups."""
        stored: List[FactRecord] = []
        for fact in facts:
            text = (fact or "").strip()
            if not text:
                continue
            stored.append(await self._facts.add_fact(session_id, text))
        return stored

    async def all_facts(self, session_id: str) -> List[FactRecord]:
        return await self._facts.list_facts(session_id)

    async def context_block(self, session_id: str) -> str:
        """Every remembered row, unconditionally, as plain text (§5.1)."""
        facts = await self.all_facts(session_id)
        if not facts:
            return ""
        lines = [f"- {fact.fact_text}" for fact in facts]
        return "Everything remembered about this session so far:\n" + "\n".join(lines)
