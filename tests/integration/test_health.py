from __future__ import annotations

from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.api.dependencies import get_retriever


def test_liveness_always_200(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_liveness_returns_request_id_header(client: TestClient):
    response = client.get("/health")
    assert "X-Request-ID" in response.headers


def test_liveness_honours_incoming_request_id(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "my-test-id"})
    assert response.headers["X-Request-ID"] == "my-test-id"


def test_readiness_200_when_index_loaded(client: TestClient):
    response = client.get("/ready")
    # The test app uses MockRetriever which always reports is_ready() = True
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_503_when_index_missing():

    class NotReadyRetriever:
        def is_ready(self):
            return False

    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: NotReadyRetriever()
    c = TestClient(app, raise_server_exceptions=False)
    response = c.get("/ready")
    assert response.status_code == 503
