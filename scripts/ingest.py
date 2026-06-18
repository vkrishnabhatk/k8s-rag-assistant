#!/usr/bin/env python3
"""CLI entry point: fetch → clean → chunk → embed → FAISS index."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from rag_assistant.config import get_settings
from rag_assistant.ingestion.chunker import chunk_all_docs
from rag_assistant.ingestion.cleaner import clean_text
from rag_assistant.ingestion.fetcher import FetchedDoc, fetch_docs
from rag_assistant.ingestion.indexer import build_index, index_exists
from rag_assistant.retrieval.embedder import Embedder

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Kubernetes docs into a FAISS vector index")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if index already exists",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="Override the default list of URLs to ingest",
    )
    return parser.parse_args()


async def _fetch(urls: list[str]) -> list[FetchedDoc]:
    return await fetch_docs(urls)


def run(args: argparse.Namespace) -> None:
    settings = get_settings()

    if not args.force and index_exists(settings.faiss_index_path, settings.faiss_metadata_path):
        log.info(
            "index_already_exists",
            index=str(settings.faiss_index_path),
            tip="use --force to re-index",
        )
        return

    urls = args.urls or settings.k8s_docs_urls
    log.info("starting_ingestion", url_count=len(urls))

    # 1. Fetch (async — keep asyncio scope minimal to avoid torch threading conflict)
    docs: list[FetchedDoc] = asyncio.run(_fetch(urls))

    # 2. Clean
    cleaned: list[FetchedDoc] = []
    for doc in docs:
        doc.raw_text = clean_text(doc.raw_text)
        cleaned.append(doc)
    log.info("cleaning_complete", docs=len(cleaned))

    # 3. Chunk
    chunks = chunk_all_docs(
        cleaned,
        chunk_size=settings.chunk_size_words,
        overlap=settings.chunk_overlap_words,
    )
    log.info("chunking_complete", chunks=len(chunks))

    # 4. Embed + index (PyTorch runs entirely outside the asyncio event loop)
    embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
    build_index(chunks, embedder, settings.faiss_index_path, settings.faiss_metadata_path)
    log.info("ingestion_complete", chunks_indexed=len(chunks))


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        log.error("ingestion_failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
