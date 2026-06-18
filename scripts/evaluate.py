#!/usr/bin/env python3
"""CLI entry point: run RAGAS evaluation against the golden dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from rag_assistant.api.dependencies import get_llm_client, get_retriever, get_validator
from rag_assistant.evaluation.ragas_runner import evaluate

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the golden dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/data/golden_dataset.json"),
        help="Path to the golden Q&A dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_report.json"),
        help="Path to write the evaluation report JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("starting_evaluation", dataset=str(args.dataset))

    retriever = get_retriever()
    if not retriever.is_ready():
        log.error("index_not_ready", tip="Run 'make ingest' first")
        sys.exit(1)

    try:
        scores = evaluate(
            dataset_path=args.dataset,
            output_path=args.output,
            retriever=retriever,
            llm_client=get_llm_client(),
            validator=get_validator(),
            top_k=args.top_k,
        )
    except ImportError as exc:
        log.error("missing_eval_deps", error=str(exc))
        sys.exit(1)
    except Exception as exc:
        log.error("evaluation_failed", error=str(exc))
        sys.exit(1)

    print("\n── RAGAS Evaluation Results ──────────────────────────")
    for metric, score in scores.items():
        print(f"  {metric:<30} {score:.4f}")
    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
