from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_assistant.ingestion.chunker import chunk_document
from rag_assistant.ingestion.fetcher import FetchedDoc


def _doc(text: str) -> FetchedDoc:
    return FetchedDoc(
        url="https://test.example/page",
        title="Test Page",
        raw_text=text,
        fetched_at=datetime.now(tz=UTC),
        content_hash="abc123",
    )


def test_empty_text_returns_no_chunks():
    chunks = chunk_document(_doc(""), chunk_size=500, overlap=50)
    assert chunks == []


def test_short_text_returns_single_chunk():
    text = " ".join(["word"] * 100)
    chunks = chunk_document(_doc(text), chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].word_count == 100


def test_overlap_preserved():
    words = [f"w{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_document(_doc(text), chunk_size=500, overlap=50)
    assert len(chunks) == 2
    # Last 50 words of chunk[0] must equal first 50 words of chunk[1]
    tail = chunks[0].text.split()[-50:]
    head = chunks[1].text.split()[:50]
    assert tail == head


def test_chunk_ids_are_deterministic():
    text = " ".join(["hello"] * 600)
    doc = _doc(text)
    chunks1 = chunk_document(doc, chunk_size=500, overlap=50)
    chunks2 = chunk_document(doc, chunk_size=500, overlap=50)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_chunk_ids_are_unique():
    text = " ".join(["hello"] * 1100)
    chunks = chunk_document(_doc(text), chunk_size=500, overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_preserves_source_metadata():
    chunks = chunk_document(_doc("word " * 10), chunk_size=500, overlap=50)
    assert chunks[0].source_url == "https://test.example/page"
    assert chunks[0].source_title == "Test Page"


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_document(_doc("word " * 10), chunk_size=0, overlap=0)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_document(_doc("word " * 600), chunk_size=500, overlap=500)
