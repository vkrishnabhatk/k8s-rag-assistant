from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceDoc(BaseModel):
    url: str
    title: str
    score: float
    chunk_index: int


class ValidationInfo(BaseModel):
    passed: bool
    guardrail_triggered: str | None
    confidence_score: float
    overlap_score: float
    message: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    validation: ValidationInfo
    latency_ms: float
    request_id: str
    model: str
    usage: dict[str, int]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
