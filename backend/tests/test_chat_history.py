"""Durable chat history.

The regression these guard: the transcript used to live only in React state, so
reloading the page wiped it while the graph and memory survived. A CEO showing
a benchmark would lose the conversation on refresh.
"""

import pytest

from app.metrics.history import ChatHistoryRepository


@pytest.fixture
async def repo(tmp_path):
    r = ChatHistoryRepository(db_path=str(tmp_path / "chat.db"))
    await r.initialize()
    return r


async def _record_turn(repo, session_id="s1", agent_id="agent1", run="run1", turn=1,
                       user="What database do we use?", answer="PostgreSQL.", metrics=None):
    await repo.record(session_id=session_id, agent_id=agent_id, run_group_id=run,
                      turn_index=turn, role="user", content=user)
    return await repo.record(session_id=session_id, agent_id=agent_id, run_group_id=run,
                             turn_index=turn, role="assistant", content=answer, metrics=metrics)


@pytest.mark.asyncio
async def test_history_survives_a_new_repository_instance(repo):
    """The point of persisting: a fresh process (new repo object, same file)
    must still see the transcript."""
    await _record_turn(repo)

    reopened = ChatHistoryRepository(db_path=repo.db_path)
    await reopened.initialize()
    rows = await reopened.list_session("s1", agent_id="agent1")

    assert len(rows) == 2
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["content"] == "PostgreSQL."


@pytest.mark.asyncio
async def test_rows_are_returned_in_insertion_order(repo):
    """Both agents answer a turn concurrently, so ordering must come from the
    autoincrement, not the timestamp (which can tie)."""
    await _record_turn(repo, run="run1", turn=1, answer="turn one answer")
    await _record_turn(repo, run="run2", turn=2, answer="turn two answer")

    rows = await repo.list_session("s1", agent_id="agent1")
    assert [r["content"] for r in rows] == [
        "What database do we use?", "turn one answer",
        "What database do we use?", "turn two answer",
    ]
    assert [r["turn_index"] for r in rows] == [1, 1, 2, 2]


@pytest.mark.asyncio
async def test_agents_are_isolated_from_each_other(repo):
    """Each panel fetches its own transcript; they must not bleed together."""
    await _record_turn(repo, agent_id="agent1", answer="agent one says")
    await _record_turn(repo, agent_id="agent2", answer="agent two says")

    a1 = await repo.list_session("s1", agent_id="agent1")
    a2 = await repo.list_session("s1", agent_id="agent2")

    assert a1[-1]["content"] == "agent one says"
    assert a2[-1]["content"] == "agent two says"
    assert all(r["agent_id"] == "agent1" for r in a1)
    assert all(r["agent_id"] == "agent2" for r in a2)


@pytest.mark.asyncio
async def test_sessions_do_not_leak_into_each_other(repo):
    await _record_turn(repo, session_id="s1")
    await _record_turn(repo, session_id="s2")

    assert len(await repo.list_session("s1")) == 2
    assert len(await repo.list_session("s2")) == 2


@pytest.mark.asyncio
async def test_metrics_ride_along_with_the_answer(repo):
    """One row restores both the chat bubble and the dashboard figures."""
    metrics = {"total_context_tokens": 4200, "cost_projected": {"gpt-4o": 0.011}}
    await _record_turn(repo, metrics=metrics)

    rows = await repo.list_session("s1", agent_id="agent1")
    assert rows[-1]["metrics"]["total_context_tokens"] == 4200
    assert rows[-1]["metrics"]["cost_projected"]["gpt-4o"] == 0.011
    # The user row has no metrics — it was never a measured turn.
    assert rows[0]["metrics"] is None


@pytest.mark.asyncio
async def test_tool_calls_round_trip(repo):
    await repo.record(
        session_id="s1", agent_id="agent2", run_group_id="run1", turn_index=1,
        role="assistant", content="done",
        tool_calls=[{"tool": "search", "latency_ms": 12.5}],
    )
    rows = await repo.list_session("s1")
    assert rows[0]["tool_calls"] == [{"tool": "search", "latency_ms": 12.5}]


@pytest.mark.asyncio
async def test_list_sessions_reports_most_recent_first(repo):
    await _record_turn(repo, session_id="older", run="r1")
    await _record_turn(repo, session_id="newer", run="r2")

    sessions = await repo.list_sessions()
    ids = [s["session_id"] for s in sessions]
    assert set(ids) == {"older", "newer"}
    assert ids[0] == "newer"  # most recently active first
    assert all(s["turns"] == 1 for s in sessions)


@pytest.mark.asyncio
async def test_clear_is_scoped_to_one_session(repo):
    await _record_turn(repo, session_id="s1")
    await _record_turn(repo, session_id="s2")

    assert await repo.clear("s1") == 2
    assert await repo.list_session("s1") == []
    assert len(await repo.list_session("s2")) == 2  # untouched

    assert await repo.clear() == 2  # clear-all
    assert await repo.list_sessions() == []


@pytest.mark.asyncio
async def test_a_corrupt_json_blob_costs_one_row_not_the_page(repo):
    """A single unreadable row must not take down the whole transcript."""
    import aiosqlite

    await _record_turn(repo)
    async with aiosqlite.connect(repo.db_path) as db:
        await db.execute("UPDATE chat_turns SET metrics = 'not-json' WHERE role = 'assistant'")
        await db.commit()

    rows = await repo.list_session("s1")
    assert len(rows) == 2
    assert rows[0]["content"] == "What database do we use?"  # intact
    assert rows[1]["metrics"] is None  # degraded, not fatal
