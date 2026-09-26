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

    r8 = client.post("/debug/replay", json={"query": "What database are we currently using?"})
    assert r8.status_code == 200
    assert "baseline" in r8.json() and "memtrace" in r8.json()


def test_chat_ingests_and_answers(client):
    response = client.post("/agent/chat", json={"message": "Project Alpha uses Redis for caching."})
    assert response.status_code == 200
    body = response.json()
    assert "Redis" in body["answer"]
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
