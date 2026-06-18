from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_assistant.ingestion.chunker import Chunk
from rag_assistant.retrieval.retriever import RetrievalResult, Retriever


def _mock_embedder(dimension: int = 384):
    embedder = MagicMock()
    embedder.dimension = dimension
    embedder.encode_query.return_value = np.random.rand(1, dimension).astype("float32")
    return embedder


def _sample_chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        text=f"Kubernetes concept {i}: pods are the smallest deployable units.",
        source_url=f"https://kubernetes.io/docs/page{i}/",
        source_title=f"Page {i}",
        word_count=10,
        chunk_index=i,
    )


def test_retriever_not_ready_when_files_missing(tmp_path):
    embedder = _mock_embedder()
    retriever = Retriever(
        index_path=tmp_path / "missing.index",
        metadata_path=tmp_path / "missing.pkl",
        embedder=embedder,
    )
    assert not retriever.is_ready()


def test_retriever_raises_when_not_ready(tmp_path):
    embedder = _mock_embedder()
    retriever = Retriever(
        index_path=tmp_path / "missing.index",
        metadata_path=tmp_path / "missing.pkl",
        embedder=embedder,
    )
    with pytest.raises(RuntimeError, match="FAISS index is not loaded"):
        retriever.search("test query")


def test_retriever_returns_top_k(tmp_path):
    import pickle

    import faiss

    dimension = 384
    n_chunks = 10
    chunks = [_sample_chunk(i) for i in range(n_chunks)]
    vectors = np.random.rand(n_chunks, dimension).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path = tmp_path / "test.index"
    meta_path = tmp_path / "test_chunks.pkl"
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    embedder = _mock_embedder(dimension)
    q = np.random.rand(1, dimension).astype("float32")
    faiss.normalize_L2(q)
    embedder.encode_query.return_value = q

    retriever = Retriever(index_path, meta_path, embedder)
    assert retriever.is_ready()

    result = retriever.search("what is a pod?", top_k=3)
    assert len(result.chunks) == 3
    assert len(result.scores) == 3
    assert result.max_score == max(result.scores)


def test_retrieval_result_max_score():
    chunks = [_sample_chunk(i) for i in range(3)]
    scores = [0.9, 0.7, 0.5]
    result = RetrievalResult(chunks=chunks, scores=scores)
    assert result.max_score == 0.9


def test_retrieval_result_empty_scores():
    result = RetrievalResult(chunks=[], scores=[])
    assert result.max_score == 0.0
