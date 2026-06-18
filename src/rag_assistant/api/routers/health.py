from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_assistant.api.dependencies import get_retriever
from rag_assistant.api.schemas import HealthResponse, ReadyResponse
from rag_assistant.retrieval.retriever import Retriever

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness probe — always returns 200 if the process is alive."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def readiness(
    retriever: Retriever = Depends(get_retriever),  # noqa: B008
) -> ReadyResponse:
    """Readiness probe — returns 200 only when the FAISS index is loaded."""
    if not retriever.is_ready():
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded. Run ingestion first.",
        )
    return ReadyResponse(status="ready")
