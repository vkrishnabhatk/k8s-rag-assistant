from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag_assistant.ingestion.fetcher import FetchedDoc


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    source_title: str
    word_count: int
    chunk_index: int


def _make_chunk_id(source_url: str, chunk_index: int) -> str:
    raw = f"{source_url}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_document(
    doc: FetchedDoc,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Sliding-window word-based chunker with configurable overlap."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    words = doc.raw_text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    idx = 0

    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if not window:
            break
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(doc.url, idx),
                text=text,
                source_url=doc.url,
                source_title=doc.title,
                word_count=len(window),
                chunk_index=idx,
            )
        )
        idx += 1
        # Stop if we've consumed all words (last window was shorter than chunk_size)
        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_all_docs(
    docs: list[FetchedDoc],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk all fetched documents and return a flat list."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
