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
