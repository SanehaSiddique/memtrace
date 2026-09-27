"""Post-response JEV graph cleaning.

The background write in `agent2/staleness.py` only checks a *newly extracted*
fact against the active facts for the same entity. That misses contradictions
that already exist in the graph from earlier turns — e.g. two facts about the
same subject that were ingested before either superseded the other, or facts
whose entity names differed slightly so the per-entity lookup missed them.

This pass closes that gap. After a turn has been answered, it takes the most
recent `window` (default 5) fact nodes for the session and asks JEV, pairwise,
whether each older node conflicts with a newer one. When JEV confirms a
conflict, the *older* node is marked stale and linked with SUPERSEDED_BY, so
the graph keeps the history explaining why the current answer is current.

Every node is classified, never deleted
--------------------------------------
Retired nodes are **marked stale and classified**, then kept. The §6.1
invariant is that stale facts are never deleted: the graph's value is the
lineage that shows *why* today's answer is today's answer, and deleting the
old node makes the graph lie about its own history. So this module has no
delete path at all — there is nothing to configure to turn one on.

Classification is a second JEV call, not a keyword match. Once a conflict is
confirmed, JEV picks *why* the old fact lost — the distinction matters to a
reader ("we changed our mind" vs "that was a typo" vs "this is a different
subject entirely"), so the classes are semantic rather than derived from the
conflict verdict itself. The class and its confidence ride along on the node
(`FactNode.stale_class`) and are surfaced in the graph snapshot, which is what
the UI groups by in the sidebar.

The whole pass is wrapped so a JEV outage can never fail a user turn: it runs
decoupled from the response path, and every failure degrades to "no cleaning
this turn" with the real reason recorded.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app.core.jev_client import JEVClient
from app.db.neo4j_driver import FactGraphStore
from app.tracing.langsmith import traced

AGENT_ID = "agent2"

# The taxonomy JEV chooses from when a fact is retired, with the one-line
# meaning sent to JEV as the option's criteria. Keys are what gets stored and
# grouped on; values are what a human reads in the sidebar.
STALE_CLASSES: Dict[str, str] = {
    "outdated": "the fact was true at the time but the situation has since changed",
    "corrected": "the newer fact fixes an error or mistake in the older one",
    "preference_change": "the subject changed their mind, choice, or stated preference",
    "duplicate": "the newer fact restates the older one with different wording",
    "scope_change": "the newer fact narrows or widens what the older one covered",
    "different_subject": "the two facts are about different subjects and only looked like a conflict",
    "unclassified": "JEV could not confidently say why the older fact was retired",
}


@dataclass
class GraphCleaningResult:
    """What one cleaning pass changed (all zeros = nothing to clean)."""

    window_examined: int = 0
    pairs_checked: int = 0
    conflicts_found: int = 0
    nodes_marked_stale: int = 0
    stale_pairs: List[Tuple[str, str]] = field(default_factory=list)  # (old_id, new_id)
    class_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.nodes_marked_stale > 0


async def _classify_conflict(
    jev: JEVClient,
    older,
    newer,
    run_group_id: str,
) -> Tuple[str, str, float]:
    """Ask JEV *why* the older fact lost. Returns (class, reason, confidence).

    Failure is not fatal: the node is still marked stale, just under
    `unclassified`. A mislabelled retirement is recoverable; a missed one is
    not, so classification never gates the marking.
    """
    res = await jev.jev_choice(
        question="Why is the older fact no longer the one to believe?",
        options=list(STALE_CLASSES.keys()),
        option_criteria=STALE_CLASSES,
        instructions=(
            "Classify why the older fact should be retired in favour of the newer one. "
            "Pick 'different_subject' if they are not actually in conflict."
        ),
        context={
            "subject": older.entity or newer.entity or "unknown",
            "older_fact": older.text,
            "newer_fact": newer.text,
        },
        call_site="staleness_check",
        agent_id=AGENT_ID,
        run_group_id=run_group_id,
    )
    if not res.ok or not res.choice:
        return "unclassified", f"JEV classification unavailable: {res.error or 'no choice'}", 0.0
    if res.choice not in STALE_CLASSES:
        return "unclassified", f"JEV returned unknown class {res.choice!r}", 0.0
    return res.choice, STALE_CLASSES[res.choice], float(res.confidence)


@traced(name="agent2.graph_cleaning")
async def clean_recent_graph(
    graph_store: FactGraphStore,
    jev: JEVClient,
    session_id: str,
    window: int = 5,
    conflict_threshold: float = 0.6,
    run_group_id: str = "",
) -> GraphCleaningResult:
    """JEV-check the newest `window` facts and stale the older of each conflict."""
    result = GraphCleaningResult()
    if window <= 0:
        return result

    recent = await graph_store.recent_facts(session_id, window)
    result.window_examined = len(recent)
    if len(recent) < 2:
        return result

    # Oldest first, so index i is always the candidate for retirement and only
    # ever compared against facts that came after it.
    ordered = sorted(recent, key=lambda f: f.created_at or "")

    for i, older in enumerate(ordered):
        for newer in ordered[i + 1 :]:
            if older.status != "active":
                # Already retired by an earlier pair in this same pass.
                break

            question = "Do these two stored facts contradict each other, so that only the more recent one should be believed?"
            context = {
                "subject": older.entity or newer.entity or "unknown",
                "older_fact": older.text,
                "newer_fact": newer.text,
            }

            res = await jev.jev_noul(
                question=question,
                context=context,
                call_site="staleness_check",
                true_meaning="yes, the two facts contradict each other and the newer one replaces the older one",
                false_meaning="no, both facts can be true at the same time or they describe different things",
                agent_id=AGENT_ID,
                run_group_id=run_group_id,
            )
            result.pairs_checked += 1

            if not res.ok:
                result.notes.append(f"jev_noul_unavailable: {res.error}")
                continue

            if not (res.value and res.probability >= conflict_threshold):
                continue

            result.conflicts_found += 1
            stale_class, reason, confidence = await _classify_conflict(jev, older, newer, run_group_id)
            await graph_store.mark_stale(
                older.id,
                newer.id,
                session_id,
                stale_class=stale_class,
                stale_reason=reason,
                stale_confidence=confidence,
            )
            result.nodes_marked_stale += 1
            result.stale_pairs.append((older.id, newer.id))
            result.class_counts[stale_class] = result.class_counts.get(stale_class, 0) + 1
            result.notes.append(
                f"cleaned[{stale_class}]: '{older.text}' -> '{newer.text}' (confidence {confidence:.2f})"
            )

            # The node is retired; no need to test it against anything later.
            break

    return result