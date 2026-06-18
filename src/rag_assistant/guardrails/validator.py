from __future__ import annotations

from dataclasses import dataclass

import structlog

from rag_assistant.config import Settings
from rag_assistant.ingestion.chunker import Chunk

log = structlog.get_logger(__name__)

_OUT_OF_CONTEXT_PHRASES = [
    "i don't have enough information",
    "i do not have enough information",
    "not in the provided context",
    "cannot find",
    "no information available",
    "not mentioned in the context",
    "the context does not",
    "context doesn't contain",
]


@dataclass
class ValidationResult:
    passed: bool
    guardrail_triggered: str | None  # "low_confidence" | "out_of_context" | "low_overlap" | None
    confidence_score: float
    overlap_score: float
    message: str | None = None


class GuardrailValidator:
    """Three-layer guardrail system for RAG response validation."""

    def __init__(self, settings: Settings) -> None:
        self._confidence_threshold = settings.retrieval_confidence_threshold
        self._overlap_threshold = settings.guardrail_overlap_threshold

    def check_retrieval_confidence(self, max_score: float) -> ValidationResult:
        """Guardrail 3 (pre-LLM): reject if top retrieval similarity is below threshold.

        Prevents wasting an LLM call when no relevant context was found.
        """
        passed = max_score >= self._confidence_threshold
        result = ValidationResult(
            passed=passed,
            guardrail_triggered=None if passed else "low_confidence",
            confidence_score=max_score,
            overlap_score=0.0,
            message=(
                None
                if passed
                else (
                    f"No sufficiently relevant context found in the Kubernetes documentation "
                    f"(best similarity: {max_score:.3f}, threshold: {self._confidence_threshold})."
                )
            ),
        )
        if not passed:
            log.info(
                "guardrail_triggered",
                guardrail="low_confidence",
                max_score=max_score,
                threshold=self._confidence_threshold,
            )
        return result

    def _check_out_of_context(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(phrase in lowered for phrase in _OUT_OF_CONTEXT_PHRASES)

    def _compute_overlap(self, answer: str, chunks: list[Chunk]) -> float:
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(c.text for c in chunks).lower().split())
        if not answer_words and not context_words:
            return 1.0
        union = answer_words | context_words
        if not union:
            return 0.0
        return len(answer_words & context_words) / len(union)

    def validate_response(
        self,
        answer: str,
        chunks: list[Chunk],
        max_score: float,
    ) -> ValidationResult:
        """Post-generation guardrails (1 and 2).

        Guardrail 1: LLM signals it cannot answer from context.
        Guardrail 2: Jaccard word overlap between answer and retrieved context.
        """
        overlap = self._compute_overlap(answer, chunks)

        # Guardrail 1: out-of-context detection
        if self._check_out_of_context(answer):
            result = ValidationResult(
                passed=False,
                guardrail_triggered="out_of_context",
                confidence_score=max_score,
                overlap_score=overlap,
                message="The model indicated it could not answer from the available context.",
            )
            log.info("guardrail_triggered", guardrail="out_of_context")
            return result

        # Guardrail 2: word overlap check
        if overlap < self._overlap_threshold:
            result = ValidationResult(
                passed=False,
                guardrail_triggered="low_overlap",
                confidence_score=max_score,
                overlap_score=overlap,
                message=(
                    f"Response appears to drift from retrieved context "
                    f"(overlap: {overlap:.3f}, threshold: {self._overlap_threshold})."
                ),
            )
            log.info(
                "guardrail_triggered",
                guardrail="low_overlap",
                overlap=overlap,
                threshold=self._overlap_threshold,
            )
            return result

        return ValidationResult(
            passed=True,
            guardrail_triggered=None,
            confidence_score=max_score,
            overlap_score=overlap,
        )
