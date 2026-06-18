from __future__ import annotations

from functools import lru_cache

from rag_assistant.config import get_settings
from rag_assistant.generation.llm_client import LLMClient
from rag_assistant.guardrails.validator import GuardrailValidator
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.retriever import Retriever


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    return Embedder(s.embedding_model, device=s.embedding_device)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    s = get_settings()
    return Retriever(
        index_path=s.faiss_index_path,
        metadata_path=s.faiss_metadata_path,
        embedder=get_embedder(),
    )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())


@lru_cache(maxsize=1)
def get_validator() -> GuardrailValidator:
    return GuardrailValidator(get_settings())
