"""End-to-end API smoke test through FastAPI's TestClient, exercising the
full ingest -> chat -> debug loop against a fresh, isolated SQLite DB."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client(tmp_path):
    original_db_path = settings.memtrace_db_path
    settings.memtrace_db_path = str(tmp_path / "api_test.db")
    with TestClient(app) as test_client:
        yield test_client
    settings.memtrace_db_path = original_db_path


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


def test_chat_answers_without_implicitly_writing_memory(client):
    response = client.post("/agent/chat", json={"message": "What can you help me with?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["memory_status"] == "skipped"
    assert body["memory_operations"] == []


def test_chat_can_explicitly_remember_and_answer(client):
    response = client.post(
        "/agent/chat",
        json={"message": "Project Alpha uses Redis for caching.", "remember": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["memory_status"] == "saved"
    assert len(body["memory_operations"]) == 1


def test_evaluate_endpoint(client):
    client.post("/memory/ingest", json={"content": "Project Alpha uses MongoDB."})
    response = client.post("/evaluate")
    assert response.status_code == 200
    body = response.json()
    assert "memtrace_accuracy" in body
    assert len(body["cases"]) == 10


def test_cost_endpoints_reflect_recorded_runs(client):
    client.post("/demo/seed")
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

    projection = client.post(
        "/cost/scale-projection",
        json={"agents": 1000, "runs_per_agent_per_day": 500, "cost_per_run": 0.08, "avoidable_pct": 22},
    ).json()
    assert projection["annual_savings"] == pytest.approx(1000 * 500 * 0.08 * 0.22 * 365)

    roi = client.get("/cost/memory-roi").json()
    stale_row = next(r for r in roi["rows"] if r["operation"] == "outdated_information")
    assert stale_row["event_count"] == 1
    assert roi["total_cost_avoided"] > 0

    summary_after = client.get("/cost/summary").json()
    assert summary_after["memory_driven_savings"] == pytest.approx(roi["total_cost_avoided"])

    monthly_timeseries = client.get("/cost/timeseries", params={"granularity": "month"}).json()
    assert len(monthly_timeseries) == 1


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
