from __future__ import annotations

import os

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

# Prevent joblib/loky from spawning parallel worker processes for tokenization.
# Without this, sentence-transformers segfaults on macOS (Python 3.13+) due to
# a conflict between loky's subprocess spawning and the Anaconda runtime.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

log = structlog.get_logger(__name__)


class Embedder:
    """Wraps SentenceTransformer; encodes texts and returns L2-normalised float32 arrays."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        log.info("loading_embedding_model", model=model_name, device=device)
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        log.info("embedding_model_loaded", dimension=self.dimension)

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts → float32 array of shape (N, dim), L2-normalised."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=64,
        )
        vectors = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → float32 array of shape (1, dim), L2-normalised."""
        return self.encode([query])
