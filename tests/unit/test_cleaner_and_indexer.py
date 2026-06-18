from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from rag_assistant.ingestion.chunker import Chunk
from rag_assistant.ingestion.cleaner import clean_text
from rag_assistant.ingestion.indexer import build_index, index_exists

# ── Cleaner ───────────────────────────────────────────────────────────────────


def test_clean_text_collapses_whitespace():
    result = clean_text("hello   world")
    assert result == "hello world"


def test_clean_text_collapses_multiple_newlines():
    result = clean_text("line1\n\n\n\nline2")
    assert "line1" in result
    assert "line2" in result
    assert "\n\n\n" not in result


def test_clean_text_strips_empty_lines():
    result = clean_text("  \nhello\n  \nworld\n  ")
    assert result == "hello\nworld"


def test_clean_text_removes_zero_width_chars():
    result = clean_text("hello​world")
    assert "​" not in result


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


def test_clean_text_normalizes_unicode():
    result = clean_text("café")  # NFC é
    assert "café" in result


# ── Indexer ───────────────────────────────────────────────────────────────────


def _make_chunks(n: int = 5) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            text=f"Kubernetes concept {i}: pods run containers in a shared namespace.",
            source_url=f"https://k8s.io/page{i}/",
            source_title=f"Page {i}",
            word_count=10,
            chunk_index=i,
        )
        for i in range(n)
    ]


class _FakeEmbedder:
    dimension = 384

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = np.random.rand(len(texts), self.dimension).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


def test_index_exists_false_when_files_missing(tmp_path: Path):
    assert not index_exists(tmp_path / "x.index", tmp_path / "x.pkl")


def test_index_exists_false_when_only_one_file(tmp_path: Path):
    idx = tmp_path / "x.index"
    idx.write_bytes(b"data")
    assert not index_exists(idx, tmp_path / "x.pkl")


def test_build_index_creates_files(tmp_path: Path):
    chunks = _make_chunks(5)
    embedder = _FakeEmbedder()
    build_index(chunks, embedder, tmp_path / "t.index", tmp_path / "t.pkl")
    assert (tmp_path / "t.index").exists()
    assert (tmp_path / "t.pkl").exists()
    assert index_exists(tmp_path / "t.index", tmp_path / "t.pkl")


def test_build_index_metadata_matches_chunks(tmp_path: Path):
    chunks = _make_chunks(5)
    embedder = _FakeEmbedder()
    build_index(chunks, embedder, tmp_path / "t.index", tmp_path / "t.pkl")
    with open(tmp_path / "t.pkl", "rb") as f:
        stored = pickle.load(f)
    assert len(stored) == 5
    assert stored[0].chunk_id == chunks[0].chunk_id


def test_build_index_faiss_vector_count(tmp_path: Path):
    n = 7
    chunks = _make_chunks(n)
    embedder = _FakeEmbedder()
    build_index(chunks, embedder, tmp_path / "t.index", tmp_path / "t.pkl")
    index = faiss.read_index(str(tmp_path / "t.index"))
    assert index.ntotal == n


def test_build_index_raises_on_empty_chunks(tmp_path: Path):
    import pytest

    embedder = _FakeEmbedder()
    with pytest.raises(ValueError, match="empty"):
        build_index([], embedder, tmp_path / "t.index", tmp_path / "t.pkl")
