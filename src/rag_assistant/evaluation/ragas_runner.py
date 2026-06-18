from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    """Load the golden Q&A dataset from a JSON file."""
    with open(path) as f:
        data: list[dict[str, Any]] = json.load(f)
    log.info("golden_dataset_loaded", path=str(path), samples=len(data))
    return data


def run_rag_pipeline_on_sample(
    sample: dict[str, Any],
    retriever: Any,
    llm_client: Any,
    validator: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the full RAG pipeline on a single golden sample.

    Returns a dict with question, answer, contexts, and ground_truth
    formatted for RAGAS evaluation.
    """
    from rag_assistant.generation.prompt_builder import build_messages

    question = sample["question"]
    result = retriever.search(question, top_k=top_k)

    # Apply guardrail 3
    conf_check = validator.check_retrieval_confidence(result.max_score)
    if not conf_check.passed:
        return {
            "question": question,
            "answer": "Insufficient context in the Kubernetes documentation.",
            "contexts": [c.text for c in result.chunks],
            "ground_truth": sample["ground_truth"],
        }

    system_blocks, messages = build_messages(question, result.chunks, result.scores)
    answer, _ = llm_client.complete(system_blocks, messages)

    return {
        "question": question,
        "answer": answer,
        "contexts": [c.text for c in result.chunks],
        "ground_truth": sample["ground_truth"],
    }


def evaluate(
    dataset_path: Path,
    output_path: Path,
    retriever: Any,
    llm_client: Any,
    validator: Any,
    top_k: int = 5,
) -> dict[str, float]:
    """Run RAGAS evaluation on the golden dataset.

    Requires the [eval] optional dependency group:
        pip install -e ".[eval]"
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation requires the [eval] extras. Install with: pip install -e '.[eval]'"
        ) from exc

    samples = load_golden_dataset(dataset_path)
    log.info("running_rag_on_golden_samples", count=len(samples))

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        log.debug("evaluating_sample", index=i, question=sample["question"][:60])
        row = run_rag_pipeline_on_sample(sample, retriever, llm_client, validator, top_k)
        rows.append(row)

    dataset = Dataset.from_list(rows)
    log.info("running_ragas_metrics")

    results = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores: dict[str, float] = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
        "context_recall": float(results["context_recall"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    log.info("evaluation_complete", scores=scores, output=str(output_path))
    return scores
