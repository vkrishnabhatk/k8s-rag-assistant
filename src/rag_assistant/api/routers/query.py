from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from rag_assistant.api.dependencies import get_llm_client, get_retriever, get_validator
from rag_assistant.api.schemas import QueryRequest, QueryResponse, SourceDoc, ValidationInfo
from rag_assistant.config import Settings, get_settings
from rag_assistant.generation.llm_client import LLMClient
from rag_assistant.generation.prompt_builder import build_messages
from rag_assistant.guardrails.validator import GuardrailValidator
from rag_assistant.retrieval.retriever import Retriever

log = structlog.get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/v1", tags=["query"])

_INSUFFICIENT_CONTEXT_ANSWER = (
    "Insufficient context: the Kubernetes documentation does not contain enough information "
    "to answer this question reliably."
)


def _build_query_response(
    *,
    answer: str,
    chunks: list[Any],
    scores: list[float],
    validation_result: Any,
    latency_ms: float,
    request_id: str,
    model: str,
    usage: dict[str, Any],
) -> QueryResponse:
    sources = [
        SourceDoc(
            url=c.source_url,
            title=c.source_title,
            score=round(s, 4),
            chunk_index=c.chunk_index,
        )
        for c, s in zip(chunks, scores, strict=False)
    ]
    return QueryResponse(
        answer=answer,
        sources=sources,
        validation=ValidationInfo(
            passed=validation_result.passed,
            guardrail_triggered=validation_result.guardrail_triggered,
            confidence_score=round(validation_result.confidence_score, 4),
            overlap_score=round(validation_result.overlap_score, 4),
            message=validation_result.message,
        ),
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
        model=model,
        usage=usage,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    retriever: Retriever = Depends(get_retriever),  # noqa: B008
    llm: LLMClient = Depends(get_llm_client),  # noqa: B008
    validator: GuardrailValidator = Depends(get_validator),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> QueryResponse:
    """Full RAG pipeline — retrieve, guard, generate, validate, respond."""
    t0 = time.perf_counter()
    request_id = getattr(request.state, "request_id", "unknown")

    # 1. Retrieve
    result = retriever.search(body.query, top_k=body.top_k)

    # 2. Guardrail 3: pre-LLM confidence check
    conf_check = validator.check_retrieval_confidence(result.max_score)
    if not conf_check.passed:
        latency_ms = (time.perf_counter() - t0) * 1000
        return _build_query_response(
            answer=_INSUFFICIENT_CONTEXT_ANSWER,
            chunks=result.chunks,
            scores=result.scores,
            validation_result=conf_check,
            latency_ms=latency_ms,
            request_id=request_id,
            model=settings.anthropic_model,
            usage={},
        )

    # 3. Build prompt
    system_blocks, messages = build_messages(body.query, result.chunks, result.scores)

    # 4. Generate
    answer, usage = llm.complete(system_blocks, messages)

    # 5. Post-generation guardrails (1 + 2)
    validation = validator.validate_response(answer, result.chunks, result.max_score)
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "query_complete", latency_ms=round(latency_ms, 2), guardrail=validation.guardrail_triggered
    )

    return _build_query_response(
        answer=answer,
        chunks=result.chunks,
        scores=result.scores,
        validation_result=validation,
        latency_ms=latency_ms,
        request_id=request_id,
        model=settings.anthropic_model,
        usage=usage,
    )


@router.post("/query/stream")
async def query_stream(
    request: Request,
    body: QueryRequest,
    retriever: Retriever = Depends(get_retriever),  # noqa: B008
    llm: LLMClient = Depends(get_llm_client),  # noqa: B008
    validator: GuardrailValidator = Depends(get_validator),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> EventSourceResponse:
    """SSE streaming — retrieval is synchronous, then LLM tokens stream to the client."""

    async def _generate() -> AsyncGenerator[dict[str, Any], None]:
        result = retriever.search(body.query, top_k=body.top_k)

        # Guardrail 3: pre-LLM
        conf_check = validator.check_retrieval_confidence(result.max_score)
        if not conf_check.passed:
            yield {"data": _INSUFFICIENT_CONTEXT_ANSWER}
            sources_payload = [
                SourceDoc(
                    url=c.source_url,
                    title=c.source_title,
                    score=round(s, 4),
                    chunk_index=c.chunk_index,
                ).model_dump()
                for c, s in zip(result.chunks, result.scores, strict=False)
            ]
            done_data = {
                "sources": sources_payload,
                "validation": {
                    "passed": conf_check.passed,
                    "guardrail_triggered": conf_check.guardrail_triggered,
                    "confidence_score": round(conf_check.confidence_score, 4),
                    "overlap_score": 0.0,
                    "message": conf_check.message,
                },
            }
            yield {"event": "done", "data": json.dumps(done_data)}
            return

        system_blocks, messages = build_messages(body.query, result.chunks, result.scores)

        full_answer_parts: list[str] = []
        for token in llm.stream(system_blocks, messages):
            full_answer_parts.append(token)
            yield {"data": token}

        full_answer = "".join(full_answer_parts)
        validation = validator.validate_response(full_answer, result.chunks, result.max_score)

        sources_payload = [
            SourceDoc(
                url=c.source_url,
                title=c.source_title,
                score=round(s, 4),
                chunk_index=c.chunk_index,
            ).model_dump()
            for c, s in zip(result.chunks, result.scores, strict=False)
        ]
        done_data = {
            "sources": sources_payload,
            "validation": {
                "passed": validation.passed,
                "guardrail_triggered": validation.guardrail_triggered,
                "confidence_score": round(validation.confidence_score, 4),
                "overlap_score": round(validation.overlap_score, 4),
                "message": validation.message,
            },
        }
        yield {"event": "done", "data": json.dumps(done_data)}

    return EventSourceResponse(_generate())
