from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import structlog

from rag_assistant.ingestion.chunker import Chunk
from rag_assistant.retrieval.embedder import Embedder

log = structlog.get_logger(__name__)


def index_exists(index_path: Path, metadata_path: Path) -> bool:
    """Return True if both the FAISS index and metadata pickle exist and are non-empty."""
    return (
        index_path.exists()
        and index_path.stat().st_size > 0
        and metadata_path.exists()
        and metadata_path.stat().st_size > 0
    )


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    index_path: Path,
    metadata_path: Path,
) -> None:
    """Embed all chunks, build an IndexFlatIP FAISS index, and persist both to disk."""
    if not chunks:
        raise ValueError("Cannot build index from empty chunk list")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("embedding_chunks", count=len(chunks))
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts, show_progress=True)

    # vectors are already L2-normalised; IndexFlatIP gives exact cosine similarity
    faiss.normalize_L2(vectors)  # idempotent safety normalise
    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(vectors)

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(
        "index_built",
        chunks=len(chunks),
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        index_size_bytes=index_path.stat().st_size,
    )
