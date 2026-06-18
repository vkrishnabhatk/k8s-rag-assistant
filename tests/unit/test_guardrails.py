from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.guardrails.validator import GuardrailValidator
from rag_assistant.ingestion.chunker import Chunk

_SETTINGS = Settings(
    anthropic_api_key="sk-ant-test",  # type: ignore[arg-type]
    retrieval_confidence_threshold=0.3,
    guardrail_overlap_threshold=0.15,
)


def _validator() -> GuardrailValidator:
    return GuardrailValidator(_SETTINGS)


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="t0",
            text="A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes.",
            source_url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
            source_title="Deployments",
            word_count=14,
            chunk_index=0,
        )
    ]


# ── Guardrail 3: retrieval confidence ─────────────────────────────────────────


def test_low_confidence_fails_guardrail_3():
    v = _validator()
    result = v.check_retrieval_confidence(0.1)
    assert not result.passed
    assert result.guardrail_triggered == "low_confidence"
    assert result.message is not None


def test_confidence_at_threshold_passes_guardrail_3():
    v = _validator()
    result = v.check_retrieval_confidence(0.3)
    assert result.passed
    assert result.guardrail_triggered is None


def test_high_confidence_passes_guardrail_3():
    v = _validator()
    result = v.check_retrieval_confidence(0.9)
    assert result.passed
    assert result.guardrail_triggered is None


# ── Guardrail 1: out-of-context detection ─────────────────────────────────────


def test_out_of_context_phrase_fails_guardrail_1():
    v = _validator()
    result = v.validate_response(
        "I don't have enough information to answer this.",
        _chunks(),
        max_score=0.8,
    )
    assert not result.passed
    assert result.guardrail_triggered == "out_of_context"


def test_variant_out_of_context_phrase_detected():
    v = _validator()
    result = v.validate_response(
        "The answer is not in the provided context.",
        _chunks(),
        max_score=0.8,
    )
    assert not result.passed
    assert result.guardrail_triggered == "out_of_context"


def test_grounded_answer_passes_guardrail_1():
    v = _validator()
    result = v.validate_response(
        "A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes.",
        _chunks(),
        max_score=0.8,
    )
    assert result.guardrail_triggered != "out_of_context"


# ── Guardrail 2: word overlap ─────────────────────────────────────────────────


def test_low_overlap_fails_guardrail_2():
    v = _validator()
    result = v.validate_response(
        "Quantum entanglement is a physics phenomenon completely unrelated to containers or pods.",
        _chunks(),
        max_score=0.8,
    )
    assert not result.passed
    assert result.guardrail_triggered == "low_overlap"


def test_sufficient_overlap_passes_guardrail_2():
    v = _validator()
    result = v.validate_response(
        "A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes.",
        _chunks(),
        max_score=0.8,
    )
    assert result.passed
    assert result.guardrail_triggered is None
    assert result.overlap_score > 0.15


def test_validation_result_overlap_score_populated():
    v = _validator()
    result = v.validate_response(
        "A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes.",
        _chunks(),
        max_score=0.8,
    )
    assert result.overlap_score > 0
