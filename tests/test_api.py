from fastapi.testclient import TestClient

from server.app import create_promptshield_app


def test_tasks_endpoint_works() -> None:
    client = TestClient(create_promptshield_app())
    response = client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 3
    assert payload[0]["task_id"] == "task_easy"


def test_baseline_endpoint_works() -> None:
    client = TestClient(create_promptshield_app())
    response = client.post("/baseline", json={"mode": "heuristic", "limit_per_task": 2})
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["overall_average"] <= 1.0
    assert len(payload["task_summaries"]) == 3
