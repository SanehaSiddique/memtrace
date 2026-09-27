"""Post-response JEV graph cleaning (agent2/graph_cleaning.py).

Uses a fake JEV so the pass is exercised deterministically without network.
"""

import pytest

from app.agent2.graph_cleaning import STALE_CLASSES, clean_recent_graph
from app.core.jev_client import ChoiceResult, NoulResult
from app.db.neo4j_driver import FactGraphStore, InMemoryFactGraph


class _FakeJEV:
    """Stands in for JEVClient.

    `conflict_pairs` drives the boolean verdicts; `classes` maps a conflicting
    pair to the class JEV would pick when asked why it lost.
    """

    def __init__(self, conflict_pairs=(), available=True, classes=None, classify_ok=True):
        self.conflict_pairs = {frozenset(p) for p in conflict_pairs}
        self.available = available
        self.classes = classes or {}
        self.classify_ok = classify_ok
        self.calls = []
        self.classify_calls = []

    async def jev_noul(self, question, context, call_site, true_meaning="", false_meaning="",
                       agent_id="", run_group_id=""):
        self.calls.append(context)
        if not self.available:
            return NoulResult(ok=False, error="jev unreachable")
        older, newer = context["older_fact"], context["newer_fact"]
        conflict = frozenset((older, newer)) in self.conflict_pairs
        return NoulResult(ok=True, value=conflict, probability=0.9 if conflict else 0.1)

    async def jev_choice(self, question, options, context, call_site,
                         option_criteria=None, instructions=None, agent_id="", run_group_id=""):
        self.classify_calls.append(context)
        if not self.classify_ok:
            return ChoiceResult(ok=False, error="classification unavailable")
        key = frozenset((context["older_fact"], context["newer_fact"]))
        choice = self.classes.get(key, "outdated")
        return ChoiceResult(ok=True, choice=choice, confidence=0.87, probabilities={choice: 0.87})


async def _seed(store, texts, entity="Acme Corp"):
    nodes = []
    for text in texts:
        nodes.append(await store.add_fact("s1", text, entity))
    return nodes


@pytest.mark.asyncio
async def test_marks_older_node_stale_and_keeps_it():
    """The older conflicting node is marked stale, never removed (§6.1)."""
    store = InMemoryFactGraph()
    older, newer = await _seed(store, ["We use MySQL", "We migrated to PostgreSQL"])
    jev = _FakeJEV(conflict_pairs=[("We use MySQL", "We migrated to PostgreSQL")])

    result = await clean_recent_graph(store, jev, "s1", window=5)

    assert result.conflicts_found == 1
    assert result.nodes_marked_stale == 1
    assert result.stale_pairs == [(older.id, newer.id)]

    facts = {f.id: f for f in await store.session_facts("s1")}
    assert len(facts) == 2, "both facts stay in the graph"
    assert facts[older.id].status == "stale"
    assert facts[older.id].superseded_by == newer.id
    assert facts[newer.id].status == "active"


@pytest.mark.asyncio
async def test_jev_classifies_why_the_older_node_lost():
    """The *why* comes from a second JEV call, and rides along on the node."""
    store = InMemoryFactGraph()
    older, newer = await _seed(store, ["We use MySQL", "We migrated to PostgreSQL"])
    jev = _FakeJEV(
        conflict_pairs=[("We use MySQL", "We migrated to PostgreSQL")],
        classes={frozenset(("We use MySQL", "We migrated to PostgreSQL")): "outdated"},
    )

    result = await clean_recent_graph(store, jev, "s1", window=5)

    assert len(jev.classify_calls) == 1, "classification is its own JEV call"
    assert result.class_counts == {"outdated": 1}

    facts = {f.id: f for f in await store.session_facts("s1")}
    assert facts[older.id].stale_class == "outdated"
    assert facts[older.id].stale_reason == STALE_CLASSES["outdated"]
    assert facts[older.id].stale_confidence == 0.87
    # The surviving node carries no classification — it is still believed.
    assert facts[newer.id].stale_class == ""


@pytest.mark.asyncio
async def test_distinct_conflicts_get_distinct_classes():
    store = InMemoryFactGraph()
    await _seed(store, [
        "Deploy target is staging",
        "Deploy target is production",
        "Billing plan is monthly",
        "Billing plan is annual",
    ])
    jev = _FakeJEV(
        conflict_pairs=[("Deploy target is staging", "Deploy target is production"),
                        ("Billing plan is monthly", "Billing plan is annual")],
        classes={
            frozenset(("Deploy target is staging", "Deploy target is production")): "outdated",
            frozenset(("Billing plan is monthly", "Billing plan is annual")): "preference_change",
        },
    )

    result = await clean_recent_graph(store, jev, "s1", window=5)

    assert result.nodes_marked_stale == 2
    assert result.class_counts == {"outdated": 1, "preference_change": 1}


@pytest.mark.asyncio
async def test_classification_outage_still_retires_the_node():
    """A failed *classification* must not block the retirement — only the
    conflict verdict is load-bearing, the label is not."""
    store = InMemoryFactGraph()
    older, newer = await _seed(store, ["We use MySQL", "We migrated to PostgreSQL"])
    jev = _FakeJEV(
        conflict_pairs=[("We use MySQL", "We migrated to PostgreSQL")],
        classify_ok=False,
    )

    result = await clean_recent_graph(store, jev, "s1", window=5)

    assert result.nodes_marked_stale == 1
    assert result.class_counts == {"unclassified": 1}
    facts = {f.id: f for f in await store.session_facts("s1")}
    assert facts[older.id].status == "stale"
    assert facts[older.id].stale_class == "unclassified"
    assert facts[older.id].stale_confidence == 0.0


@pytest.mark.asyncio
async def test_no_conflict_leaves_graph_untouched():
    store = InMemoryFactGraph()
    await _seed(store, ["Acme is a lead", "Acme is in fintech", "Acme has 50 employees"])
    jev = _FakeJEV()  # nothing conflicts

    result = await clean_recent_graph(store, jev, "s1", window=5)

    assert result.pairs_checked == 3
    assert result.conflicts_found == 0
    assert result.nodes_marked_stale == 0
    assert not result.changed
    assert len(await store.session_facts("s1")) == 3


@pytest.mark.asyncio
async def test_jev_outage_never_marks_anything_stale():
    """A JEV outage must not silently clean the graph (no false retirements)."""
    store = InMemoryFactGraph()
    await _seed(store, ["We use MySQL", "We migrated to PostgreSQL"])
    jev = _FakeJEV(available=False)

    result = await clean_recent_graph(store, jev, "s1", window=5)

    assert result.conflicts_found == 0
    assert result.nodes_marked_stale == 0
    assert any("jev_noul_unavailable" in n for n in result.notes)
    assert len(await store.session_facts("s1")) == 2


@pytest.mark.asyncio
async def test_only_inspects_the_most_recent_window():
    """With window=2 only the two newest facts are compared."""
    store = InMemoryFactGraph()
    old = await store.add_fact("s1", "Ancient MySQL fact", "Acme")
    a = await store.add_fact("s1", "We use MySQL", "Acme")
    b = await store.add_fact("s1", "We migrated to PostgreSQL", "Acme")
    jev = _FakeJEV(conflict_pairs=[("We use MySQL", "We migrated to PostgreSQL")])

    result = await clean_recent_graph(store, jev, "s1", window=2)

    assert result.window_examined == 2
    assert result.pairs_checked == 1
    facts = {f.id: f for f in await store.session_facts("s1")}
    assert facts[old.id].status == "active"  # outside the window, untouched
    assert facts[a.id].status == "stale"
    assert facts[b.id].status == "active"


@pytest.mark.asyncio
async def test_below_confidence_threshold_is_not_a_conflict():
    store = InMemoryFactGraph()
    older, newer = await _seed(store, ["We use MySQL", "We migrated to PostgreSQL"])

    class _UnsureJEV(_FakeJEV):
        async def jev_noul(self, *args, **kwargs):
            # "true", but JEV is not confident enough to act on it.
            return NoulResult(ok=True, value=True, probability=0.4)

    result = await clean_recent_graph(store, _UnsureJEV(), "s1", window=5, conflict_threshold=0.6)

    assert result.conflicts_found == 0
    facts = {f.id: f for f in await store.session_facts("s1")}
    assert facts[older.id].status == "active"
    assert facts[newer.id].status == "active"


@pytest.mark.asyncio
async def test_short_window_and_empty_graph_are_no_ops():
    store = InMemoryFactGraph()
    jev = _FakeJEV()

    empty = await clean_recent_graph(store, jev, "s1", window=5)
    assert empty.window_examined == 0 and empty.pairs_checked == 0

    await store.add_fact("s1", "Only one fact", "Acme")
    single = await clean_recent_graph(store, jev, "s1", window=5)
    assert single.window_examined == 1 and single.pairs_checked == 0

    disabled = await clean_recent_graph(store, jev, "s1", window=0)
    assert disabled.window_examined == 0

@pytest.mark.asyncio
async def test_there_is_no_delete_path_in_the_cleaning_pass():
    """The pass marks and classifies; it never removes. Guards the §6.1
    invariant structurally, not just by convention."""
    import inspect

    from app.agent2 import graph_cleaning

    assert not hasattr(graph_cleaning.GraphCleaningResult(), "nodes_deleted")
    assert "delete_superseded" not in inspect.signature(graph_cleaning.clean_recent_graph).parameters
    assert not hasattr(FactGraphStore, "delete_fact")
    assert not hasattr(InMemoryFactGraph, "delete_fact")


@pytest.mark.asyncio
async def test_graph_snapshot_exposes_classifications_for_the_sidebar():
    store = InMemoryFactGraph()
    older, newer = await _seed(store, ["We use MySQL", "We migrated to PostgreSQL"])
    jev = _FakeJEV(
        conflict_pairs=[("We use MySQL", "We migrated to PostgreSQL")],
        classes={frozenset(("We use MySQL", "We migrated to PostgreSQL")): "corrected"},
    )
    await clean_recent_graph(store, jev, "s1", window=5)

    snapshot = await store.graph_snapshot("s1")
    by_id = {n["id"]: n for n in snapshot["nodes"]}
    assert by_id[older.id]["stale_class"] == "corrected"
    assert by_id[older.id]["stale_confidence"] == 0.87
    assert by_id[newer.id]["stale_class"] == ""
    assert len([n for n in snapshot["nodes"] if n["label"] == "Fact"]) == 2
