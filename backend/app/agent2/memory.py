"""Agent2 memory: STM + Neo4j Graph LTM (docs/IMPLEMENTATION.md §6.1).

STM: plain chat-history list for the session (identical to Agent1 so the test
is fair).

LTM: the Neo4j fact graph (`FactGraphStore`).
  * Only ACTIVE facts are fetched for context (`find_active_facts` / Cypher
    `MATCH (f:Fact {session_id: $session_id, status: 'active'})`).
  * Stale facts remain in the graph linked via `SUPERSEDED_BY`, but they NEVER
    enter the LLM's context window. This prevents contradictions from
    accumulating and keeps prompt token count lean.
"""

from typing import Any, Dict, List

from app.db.neo4j_driver import STATUS_ACTIVE, FactGraphStore, FactNode


class Agent2Memory:
    """STM (in-process history) + LTM (Neo4j fact graph) for session memory."""

    def __init__(self, graph_store: FactGraphStore) -> None:
        self._graph = graph_store
        self._stm: Dict[str, List[dict]] = {}  # session_id -> [{"role", "content"}]

    @property
    def graph_store(self) -> FactGraphStore:
        return self._graph

    # -- STM --------------------------------------------------------------------

    def history(self, session_id: str) -> List[dict]:
        return list(self._stm.get(session_id, []))

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        if user_message:
            self._stm.setdefault(session_id, []).append({"role": "user", "content": user_message})
        if assistant_message:
            self._stm.setdefault(session_id, []).append({"role": "assistant", "content": assistant_message})

    # -- LTM (Active facts only) ------------------------------------------------

    async def get_active_facts(self, session_id: str) -> List[FactNode]:
        """Fetch all currently active facts for this session (§6.1)."""
        all_facts = await self._graph.session_facts(session_id)
        return [f for f in all_facts if f.status == STATUS_ACTIVE]

    async def context_block(self, session_id: str) -> str:
        """Render active graph memories only. Contradictions/stale facts are excluded."""
        active = await self.get_active_facts(session_id)
        if not active:
            return ""
        lines: List[str] = []
        for fact in active:
            prefix = f"[{fact.entity}] " if fact.entity else ""
            lines.append(f"- {prefix}{fact.text}")
        return "Active memories for this session (stale entries resolved):\n" + "\n".join(lines)

    async def graph_snapshot(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Full graph snapshot (active + stale facts and edges) for GraphView."""
        return await self._graph.graph_snapshot(session_id)
