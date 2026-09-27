"""Agent2 JEV Staleness / Contradiction Resolution (docs/IMPLEMENTATION.md §6.4).

When new candidate facts are extracted for a session:
  1. Identifies the entity each fact pertains to.
  2. Queries Neo4j for existing active `:Fact` nodes under the same entity.
  3. For each existing active fact, calls `jev_noul` ("Does this new statement
     contradict or replace the existing fact?").
  4. If JEV confirms contradiction:
       old_fact.status = 'stale'
       new_fact.status = 'active'
       (old_fact)-[:SUPERSEDED_BY]->(new_fact)
  5. If JEV denies contradiction: both stay active.
  6. If JEV is unavailable: rule-based semantic heuristic fallback.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.jev_client import JEVClient
from app.db.neo4j_driver import FactGraphStore, FactNode
from app.tracing.langsmith import traced


@dataclass
class StalenessResolution:
    """Outcome of resolving extracted facts against the Neo4j graph."""

    new_facts_added: int = 0
    facts_superseded: int = 0
    stale_pairs: List[Tuple[str, str]] = field(default_factory=list)  # (old_id, new_id)
    notes: List[str] = field(default_factory=list)


def _heuristic_contradicts(old_text: str, new_text: str) -> bool:
    """Fallback when JEV endpoint is unreachable: detect common replacement patterns."""
    old_lower = old_text.lower()
    new_lower = new_text.lower()
    if old_lower == new_lower:
        return False

    # Common replacement pairs/keywords
    known_opposites = [
        ("java", "python"),
        ("mongodb", "postgres"),
        ("postgresql", "mongodb"),
        ("aws", "gcp"),
        ("azure", "aws"),
        ("q1", "q2"),
        ("q2", "q3"),
        ("q3", "q4"),
        ("remote", "onsite"),
        ("hybrid", "remote"),
    ]
    for w1, w2 in known_opposites:
        if (w1 in old_lower and w2 in new_lower) or (w2 in old_lower and w1 in new_lower):
            return True

    # Replacement indicator phrases
    replace_markers = ["moved to", "switched to", "changed to", "replaced with", "no longer", "instead of", "prefers", "now uses"]
    if any(marker in new_lower for marker in replace_markers):
        return True

    return False


@traced(name="agent2.resolve_staleness")
async def resolve_and_save_facts(
    graph_store: FactGraphStore,
    jev: JEVClient,
    session_id: str,
    extracted_facts: List[Tuple[str, str]],  # [(fact_text, entity_name), ...]
    run_group_id: str = "",
) -> StalenessResolution:
    """Evaluate contradictions against active Neo4j facts and update graph (§6.4)."""
    resolution = StalenessResolution()

    for fact_text, entity in extracted_facts:
        fact_text = (fact_text or "").strip()
        entity = (entity or "General").strip()
        if not fact_text:
            continue

        # Look up existing active facts for this entity
        active_facts = await graph_store.find_active_facts(session_id, entity)
        superseded_any = False

        # Add the new fact first as active
        new_node = await graph_store.add_fact(session_id, fact_text, entity)
        resolution.new_facts_added += 1

        for old_fact in active_facts:
            # Check contradiction via JEV boolean primitive (noul)
            question = "Does this new statement contradict or replace the existing fact?"
            context = {
                "entity": entity,
                "existing_fact": old_fact.text,
                "new_fact": fact_text,
            }

            res = await jev.jev_noul(
                question=question,
                context=context,
                call_site="staleness_check",
                true_meaning="yes, the new fact replaces or contradicts the existing fact",
                false_meaning="no, both facts can be simultaneously true or describe different aspects",
                agent_id="agent2",
                run_group_id=run_group_id,
            )

            is_contradiction = False
            if res.ok:
                is_contradiction = bool(res.value and res.probability >= 0.55)
            else:
                # Fall back to heuristic rule check if JEV is unavailable
                is_contradiction = _heuristic_contradicts(old_fact.text, fact_text)
                resolution.notes.append(f"jev_noul_fallback_used: {res.error}")

            if is_contradiction:
                await graph_store.mark_stale(old_fact.id, new_node.id, session_id)
                resolution.facts_superseded += 1
                resolution.stale_pairs.append((old_fact.id, new_node.id))
                resolution.notes.append(f"superseded: '{old_fact.text}' -> '{fact_text}'")
                superseded_any = True

    return resolution
