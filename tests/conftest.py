from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.api.dependencies import (
    get_llm_client,
    get_retriever,
)
from rag_assistant.config import Settings
from rag_assistant.ingestion.chunker import Chunk
from rag_assistant.retrieval.retriever import RetrievalResult

# ── Settings ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test-key",  # type: ignore[arg-type]
        faiss_index_path=Path("tests/data/test.index"),
        faiss_metadata_path=Path("tests/data/test_chunks.pkl"),
        rate_limit_per_minute=1000,
        retrieval_confidence_threshold=0.3,
        guardrail_overlap_threshold=0.15,
    )


# ── Sample data ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_chunks() -> list[Chunk]:
    base_url = "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/"
    texts = [
        "A Deployment provides declarative updates for Pods and ReplicaSets. "
        "You describe a desired state in a Deployment, and the Deployment Controller changes "
        "the actual state to the desired state at a controlled rate.",
        "When you create a Deployment, a ReplicaSet is created to manage the Pods. "
        "The ReplicaSet ensures that the specified number of Pod replicas are running at any time.",
        "Rolling updates allow Deployments to take place with zero downtime by incrementally "
        "updating Pod instances with new ones. The new Pods will be scheduled on Nodes with "
        "available resources.",
        "You can pause a Deployment to apply multiple fixes and then resume it to start a new rollout. "
        "This lets you apply multiple updates in the meantime without triggering unnecessary rollouts.",
        "The .spec.strategy.type field specifies the strategy used to replace old Pods by new ones. "
        "RollingUpdate is the default strategy. Recreate is an alternative strategy.",
    ]
    return [
        Chunk(
            chunk_id=f"test{i:04d}",
            text=t,
            source_url=base_url,
            source_title="Deployments",
            word_count=len(t.split()),
            chunk_index=i,
        )
        for i, t in enumerate(texts)
    ]


# ── Mock LLM client ───────────────────────────────────────────────────────────


class MockLLMClient:
    """Canned LLM client for tests — no actual API calls."""

    DEFAULT_ANSWER = (
        "A Deployment provides declarative updates for Pods and ReplicaSets. "
        "You describe a desired state in a Deployment, and the Deployment Controller changes "
        "the actual state to the desired state at a controlled rate."
    )

    def complete(self, system: list[dict], messages: list[dict]) -> tuple[str, dict]:
        return self.DEFAULT_ANSWER, {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def stream(self, system: list[dict], messages: list[dict]) -> Generator[str, None, None]:
        words = self.DEFAULT_ANSWER.split()
        for word in words:
            yield word + " "


# ── Mock Retriever ────────────────────────────────────────────────────────────


class MockRetriever:
    """Returns fixed chunks with configurable scores."""

    def __init__(self, chunks: list[Chunk], max_score: float = 0.85) -> None:
        self._chunks = chunks
        self._max_score = max_score

    def is_ready(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        n = min(top_k, len(self._chunks))
        scores = [self._max_score - i * 0.05 for i in range(n)]
        return RetrievalResult(chunks=self._chunks[:n], scores=scores)


class MockRetrieverLowConfidence:
    """Returns chunks with a score below the confidence threshold."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def is_ready(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        return RetrievalResult(chunks=self._chunks[:1], scores=[0.1])


# ── FastAPI test app ──────────────────────────────────────────────────────────


@pytest.fixture
def app(sample_chunks: list[Chunk]) -> FastAPI:
    application = create_app()
    mock_retriever = MockRetriever(sample_chunks)
    mock_llm = MockLLMClient()
    application.dependency_overrides[get_retriever] = lambda: mock_retriever
    application.dependency_overrides[get_llm_client] = lambda: mock_llm
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)
