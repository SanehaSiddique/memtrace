import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client(tmp_path):
    original_db_path = settings.memtrace_db_path
    settings.memtrace_db_path = str(tmp_path / "comparison_api_test.db")
    with TestClient(app) as test_client:
        yield test_client
    settings.memtrace_db_path = original_db_path


def test_comparison_api_endpoints(client):
    # 1. Reset endpoint
    res = client.post("/api/chat/reset?session_id=test_sess")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # 2. Metrics endpoint
    res = client.get("/api/metrics/test_sess")
    assert res.status_code == 200
    data = res.json()
    assert "turns" in data
    assert "summary" in data

    # 3. Graph snapshot endpoint
    res = client.get("/api/graph/test_sess")
    assert res.status_code == 200
    graph = res.json()
    assert "nodes" in graph
    assert "edges" in graph

    # 4. Chat comparison POST fallback endpoint
    res = client.post(
        "/api/chat/compare",
        json={"message": "What database are we using?", "session_id": "test_sess"},
    )
    assert res.status_code == 200
    comp = res.json()
    assert "run_group_id" in comp
    assert "agent1" in comp
    assert "agent2" in comp
    assert "summary" in comp
