from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import structlog

from rag_assistant.ingestion.chunker import Chunk
from rag_assistant.retrieval.embedder import Embedder

log = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    scores: list[float]
    max_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.max_score = max(self.scores) if self.scores else 0.0


class Retriever:
    """Loads a FAISS index from disk and performs top-k similarity search."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedder: Embedder,
    ) -> None:
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._embedder = embedder
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self._index_path.exists() or not self._metadata_path.exists():
            log.warning(
                "index_files_missing",
                index=str(self._index_path),
                metadata=str(self._metadata_path),
            )
            return
        self._index = faiss.read_index(str(self._index_path))  # type: ignore[assignment]
        with open(self._metadata_path, "rb") as f:
            self._chunks = pickle.load(f)
        assert self._index is not None
        log.info(
            "index_loaded",
            vectors=self._index.ntotal,
            chunks=len(self._chunks),
        )

    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self.is_ready():
            raise RuntimeError("FAISS index is not loaded. Run ingestion first.")

        query_vec = self._embedder.encode_query(query)
        # query_vec is already L2-normalised; IndexFlatIP gives cosine similarity
        scores_arr, indices_arr = self._index.search(query_vec, top_k)  # type: ignore[union-attr]

        chunks: list[Chunk] = []
        scores: list[float] = []
        for score, idx in zip(scores_arr[0], indices_arr[0], strict=False):
            if idx == -1:  # FAISS padding when k > index size
                continue
            chunks.append(self._chunks[idx])
            scores.append(float(score))

        log.debug("search_complete", query=query[:60], top_k=top_k, results=len(chunks))
        return RetrievalResult(chunks=chunks, scores=scores)
