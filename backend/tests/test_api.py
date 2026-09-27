"""End-to-end API smoke test through FastAPI's TestClient, exercising the
full ingest -> chat -> debug loop against a fresh, isolated SQLite DB.

These assertions describe the *deterministic* extraction/consolidation rules, so
the fixture pins the LLM client to the offline Mock. Without that, a real key in
`.env` (GROQ_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY) makes the suite
non-hermetic: live extraction legitimately produces different candidates and
counts than the rule-based path, and the suite used to pass only by accident
whenever the configured free provider was rate-limiting into the fallback.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.llm.counted import CountedLLMClient
from app.llm.factory import get_llm_client
from app.llm.interface import BaseLLMClient, ChatCompletion, ToolCall
from app.main import app

# Provider keys that would otherwise promote the app off the offline Mock client.
_LLM_KEY_FIELDS = ("groq_api_key", "openrouter_api_key", "openai_api_key")


@pytest.fixture
def client(tmp_path):
    original_db_path = settings.memtrace_db_path
    original_keys = {field: getattr(settings, field) for field in _LLM_KEY_FIELDS}

    settings.memtrace_db_path = str(tmp_path / "api_test.db")
    for field in _LLM_KEY_FIELDS:
        setattr(settings, field, None)

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        settings.memtrace_db_path = original_db_path
        for field, value in original_keys.items():
            setattr(settings, field, value)


def test_live_llm_client_is_counted_and_cached(monkeypatch):
    # `_env_file=None` blocks the .env *file*, but real environment variables
    # still take precedence in pydantic-settings, so clear them too. Each case
    # then selects its provider purely from the explicit arguments.
    for field in _LLM_KEY_FIELDS:
        monkeypatch.delenv(field.upper(), raising=False)

    groq_settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        groq_model="openai/gpt-oss-120b",
    )
    groq_client = get_llm_client(groq_settings)
    assert isinstance(groq_client, CountedLLMClient)
    assert groq_client.model_name == "openai/gpt-oss-120b"
    assert groq_client.inner.provider == "groq"

    # OpenAI takes priority when configured alongside Groq.
    openai_settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
    )
    openai_client = get_llm_client(openai_settings)
    assert openai_client.model_name == "gpt-4o-mini"
    assert openai_client.inner.provider == "openai-compatible"

    # Groq remains ahead of OpenRouter when OpenAI is not configured.
    both_settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="test-key",
    )
    assert get_llm_client(both_settings).inner.provider == "groq"

    openrouter_settings = Settings(
        _env_file=None,
        groq_api_key=None,
        openrouter_api_key="test-key",
        openrouter_model="nvidia/nemotron-3-super-120b-a12b:free",
    )
    openrouter_client = get_llm_client(openrouter_settings)
    assert openrouter_client.model_name == "nvidia/nemotron-3-super-120b-a12b:free"
    assert openrouter_client.inner.provider == "openrouter"

    # With no provider key at all the app stays on the offline Mock.
    assert get_llm_client(Settings(_env_file=None)).__class__.__name__ == "MockLLMClient"


@pytest.mark.asyncio
async def test_counted_client_forwards_tool_calling_chat_completions():
    """`CountedLLMClient` must forward `chat_completion` to the wrapped provider.

    Both comparison agents issue tool calls through the counted wrapper. The
    wrapper used to inherit `BaseLLMClient`'s always-raising default, so every
    tool-calling turn failed with "does not support tool-calling chat
    completions" regardless of which live provider was configured.
    """
    sent: dict = {}

    class _FakeProvider(BaseLLMClient):
        is_live = True
        embeddings_are_local = True
        provider = "fake"
        model_name = "fake-model"
        provider_call_count = 0
        rate_limit_events = 0

        async def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
            return "unused"

        async def embed(self, text: str) -> list[float]:
            return [0.0]

        async def chat_completion(self, messages, tools=None, temperature=0.0, model=None):
            self.provider_call_count += 1
            sent["tools"] = tools
            return ChatCompletion(
                content="",
                tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"location":"Paris"}')],
                usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
                model="fake-model",
            )

    provider = _FakeProvider()
    counted = CountedLLMClient(provider, cache_enabled=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]
    messages = [{"role": "user", "content": "Weather in Paris?"}]

    first = await counted.chat_completion(messages, tools=tools)
    assert [call.name for call in first.tool_calls] == ["get_weather"]
    assert sent["tools"] == tools
    assert first.model == "fake-model"

    # Accounting and caching both flow through the wrapper.
    snapshot = counted.snapshot()
    assert snapshot["chat_calls"] == 1
    assert snapshot["chat_provider_calls"] == 1
    assert snapshot["tokens_in"] == 11
    assert snapshot["tokens_out"] == 5

    second = await counted.chat_completion(messages, tools=tools)
    assert [call.name for call in second.tool_calls] == ["get_weather"]
    assert counted.snapshot()["chat_cache_hits"] == 1
    assert provider.provider_call_count == 1  # served from cache, no extra HTTP


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_then_query_then_debug(client):
    r1 = client.post("/memory/ingest", json={"content": "Project Alpha uses MongoDB."})
    assert r1.status_code == 200
    assert r1.json()["operations"][0]["operation"] == "ADD"

    r2 = client.post(
        "/memory/ingest",
        json={"content": "We migrated from MongoDB to PostgreSQL because relational querying became important."},
    )
    assert r2.status_code == 200
    assert r2.json()["operations"][0]["operation"] == "UPDATE"
    postgres_memory_id = r2.json()["operations"][0]["memory_id"]

    r3 = client.post("/memory/query", json={"query": "What database are we currently using?"})
    assert r3.status_code == 200
    body = r3.json()
    assert "PostgreSQL" in body["answer"]
    assert any(sm["memory"]["object"] == "PostgreSQL" for sm in body["selected_memories"])
    assert any(em["memory"]["object"] == "MongoDB" for em in body["excluded_memories"])

    r4 = client.get(f"/memory/{postgres_memory_id}")
    assert r4.status_code == 200
    assert r4.json()["object"] == "PostgreSQL"

    r5 = client.get(f"/memory/{postgres_memory_id}/history")
    assert r5.status_code == 200
    assert len(r5.json()) == 2

    r6 = client.get(f"/memory/{postgres_memory_id}/graph")
    assert r6.status_code == 200
    assert len(r6.json()["relationships"]) >= 1

    r7 = client.post("/debug/query", json={"query": "What database are we currently using?"})
    assert r7.status_code == 200
    debug_body = r7.json()
    assert "PostgreSQL" in debug_body["answer"]
    assert any(em["memory"]["object"] == "MongoDB" for em in debug_body["excluded"])
    assert debug_body["run_savings"] > 0
    assert debug_body["run_savings"] == pytest.approx(debug_body["run_baseline_cost"] - debug_body["run_optimized_cost"])

    r8 = client.post("/debug/replay", json={"query": "What database are we currently using?"})
    assert r8.status_code == 200
    assert "baseline" in r8.json() and "memtrace" in r8.json()


def test_chat_ingests_and_answers(client):
    response = client.post("/agent/chat", json={"message": "Project Alpha uses Redis for caching."})
    assert response.status_code == 200
    body = response.json()
    assert "Redis" in body["answer"]
    assert len(body["memory_operations"]) == 1


def test_cost_endpoints_reflect_recorded_runs(client):
    # Two real ingests that contradict each other, so the run below has genuine
    # stale context to exclude. These assertions are about the cost service
    # reporting what actually happened, not about a seeded fixture.
    client.post("/memory/ingest", json={"content": "The team uses MySQL as the primary datastore."})
    client.post(
        "/memory/ingest",
        json={"content": "We migrated the primary datastore from MySQL to PostgreSQL."},
    )
    client.post("/memory/query", json={"query": "What database are we currently using?"})

    summary = client.get("/cost/summary").json()
    assert summary["total_runs"] == 1
    assert summary["lifetime_savings"] > 0
    assert "not live provider billing data" in summary["data_source_note"]

    timeseries = client.get("/cost/timeseries").json()
    assert len(timeseries) == 1

    leaks = client.get("/cost/leaks").json()
    assert leaks[0]["category"] == "outdated_information"

    runs = client.get("/cost/runs").json()
    assert len(runs) == 1

    roi = client.get("/cost/memory-roi").json()
    stale_row = next(r for r in roi["rows"] if r["operation"] == "outdated_information")
    assert stale_row["event_count"] >= 1
    assert roi["total_cost_avoided"] > 0

    summary_after = client.get("/cost/summary").json()
    assert summary_after["memory_driven_savings"] == pytest.approx(roi["total_cost_avoided"])

    monthly_timeseries = client.get("/cost/timeseries", params={"granularity": "month"}).json()
    assert len(monthly_timeseries) == 1


def test_revenue_projection_is_measured_from_recorded_runs(client):
    """The CEO projection must price real recorded token volumes, so the run
    recorded above is what drives the numbers here."""
    client.post("/memory/ingest", json={"content": "The team uses MySQL as the primary datastore."})
    client.post(
        "/memory/ingest",
        json={"content": "We migrated the primary datastore from MySQL to PostgreSQL."},
    )
    client.post("/memory/query", json={"query": "What database are we currently using?"})

    body = client.post(
        "/cost/revenue-projection",
        json={
            "agents": 10,
            "runs_per_agent_per_day": 10,
            "current_model": "gpt-4o",
            "revenue_per_agent_month": 500,
        },
    ).json()

    assert body["profile"]["is_measured"] is True
    assert body["profile"]["measured_runs"] == 1
    # Fewer tokens sent with MEMTRACE than the retriever surfaced.
    assert body["profile"]["optimized_input_tokens"] < body["profile"]["baseline_input_tokens"]
    assert body["rows"][0]["annual_with_memtrace"] > 0
    # The catalog is reachable so the UI's model picker is never empty.
    assert client.get("/cost/models").status_code == 200


def test_seed_and_evaluate_surfaces_are_gone(client):
    """No seeding or canned-evaluation surface: the CEO view must only ever
    show figures produced by real agent activity.

    Asserted as "not 200" rather than "== 404" because the SPA catch-all answers
    unmatched paths, so a removed POST route can surface as 405 (path exists for
    GET) rather than 404. Either way, the handler is gone — 200 would mean the
    seed still ran.
    """
    assert client.post("/demo/seed").status_code != 200
    assert client.post("/evaluate").status_code != 200
    assert client.post("/cost/scale-projection", json={}).status_code != 200


def test_memory_timeline_and_graph_full_endpoints(client):
    client.post("/memory/ingest", json={"content": "Project Alpha uses MongoDB."})
    client.post(
        "/memory/ingest",
        json={"content": "We migrated from MongoDB to PostgreSQL because relational querying became important."},
    )

    timeline = client.get("/memory/timeline").json()
    assert len(timeline) == 2
    assert timeline[0]["object"] == "MongoDB"  # oldest first
    assert timeline[1]["object"] == "PostgreSQL"

    graph = client.get("/memory/graph-full").json()
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) >= 2  # CAUSED_BY x2 + REPLACED_BY/SUPERSEDES

    postgres_id = timeline[1]["id"]
    cost_impact = client.get(f"/memory/{postgres_id}/cost-impact").json()
    assert cost_impact["memory_id"] == postgres_id
    assert cost_impact["cost_avoided"] == 0  # nothing has excluded it yet

    mongo_id = timeline[0]["id"]
    client.post("/memory/query", json={"query": "What database are we currently using?"})
    mongo_impact = client.get(f"/memory/{mongo_id}/cost-impact").json()
    assert mongo_impact["cost_avoided"] > 0  # MongoDB gets excluded once queried
