from __future__ import annotations

from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.api.dependencies import get_llm_client, get_retriever
from rag_assistant.ingestion.chunker import Chunk
from tests.conftest import MockLLMClient, MockRetriever, MockRetrieverLowConfidence

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(chunks: list[Chunk], max_score: float = 0.85) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: MockRetriever(chunks, max_score=max_score)
    app.dependency_overrides[get_llm_client] = lambda: MockLLMClient()
    return TestClient(app, raise_server_exceptions=False)


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            text="A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes.",
            source_url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
            source_title="Deployments",
            word_count=14,
            chunk_index=i,
        )
        for i in range(5)
    ]


# ── POST /v1/query ─────────────────────────────────────────────────────────────


def test_query_returns_200_with_valid_body():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "What is a Kubernetes Deployment?"})
    assert r.status_code == 200


def test_query_response_has_required_fields():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "What is a Kubernetes Deployment?"})
    body = r.json()
    assert "answer" in body
    assert "sources" in body
    assert "validation" in body
    assert "latency_ms" in body
    assert "request_id" in body
    assert "model" in body


def test_query_validation_field_present_and_typed():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "What is a Kubernetes Deployment?"})
    v = r.json()["validation"]
    assert isinstance(v["passed"], bool)
    assert "guardrail_triggered" in v
    assert "confidence_score" in v
    assert "overlap_score" in v


def test_query_sources_populated():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "What is a Kubernetes Deployment?"})
    sources = r.json()["sources"]
    assert len(sources) > 0
    assert "url" in sources[0]
    assert "score" in sources[0]


def test_query_missing_body_returns_422():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={})
    assert r.status_code == 422


def test_query_short_query_returns_422():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "hi"})
    assert r.status_code == 422


def test_query_guardrail_3_no_llm_call():
    """When retrieval confidence is low, the LLM must not be called."""
    chunks = _chunks()
    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: MockRetrieverLowConfidence(chunks)

    call_count = {"n": 0}

    class CountingLLMClient:
        def complete(self, system, messages):
            call_count["n"] += 1
            return "should not be called", {}

        def stream(self, system, messages):
            call_count["n"] += 1
            yield "should not be called"

    app.dependency_overrides[get_llm_client] = lambda: CountingLLMClient()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/v1/query", json={"query": "What is quantum entanglement?"})
    assert r.status_code == 200
    assert call_count["n"] == 0
    body = r.json()
    assert "Insufficient context" in body["answer"]
    assert body["validation"]["guardrail_triggered"] == "low_confidence"


def test_query_request_id_echoed_in_header():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "What is a Pod?"}, headers={"X-Request-ID": "my-id"})
    assert r.headers.get("X-Request-ID") == "my-id"


def test_query_request_id_generated_if_absent():
    c = _make_client(_chunks())
    r = c.post("/v1/query", json={"query": "What is a Pod?"})
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) > 0


def test_query_invalid_content_type_returns_422():
    c = _make_client(_chunks())
    r = c.post("/v1/query", data="not json", headers={"Content-Type": "text/plain"})
    assert r.status_code in (415, 422)


# ── POST /v1/query/stream ─────────────────────────────────────────────────────
# sse-starlette 2.x uses anyio TaskGroups internally.
# We test the endpoint is routable and returns the right content-type via sync TestClient
# (which reads the response before the SSE generator runs fully — that is sufficient
# to confirm the route is wired and the content-type header is correct).


def test_query_stream_returns_event_stream_content_type():
    c = _make_client(_chunks())
    # Don't iterate the SSE body — just verify the endpoint is routable
    r = c.post("/v1/query/stream", json={"query": "What is a Pod?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")


def test_query_stream_missing_body_returns_422():
    c = _make_client(_chunks())
    r = c.post("/v1/query/stream", json={})
    assert r.status_code == 422
