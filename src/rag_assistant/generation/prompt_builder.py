from __future__ import annotations

from typing import Any

from rag_assistant.ingestion.chunker import Chunk

SYSTEM_PROMPT = """\
You are a Kubernetes documentation assistant. Your sole purpose is to answer \
questions about Kubernetes using only the context provided below.

Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- If the answer is not in the context, respond with exactly: \
"I don't have enough information to answer this from the Kubernetes documentation."
- Cite the source URL(s) when you reference specific information.
- Be concise and technically precise.
- Do not speculate or infer beyond what the context explicitly states.\
"""


def _format_context(chunks: list[Chunk], scores: list[float]) -> str:
    parts: list[str] = []
    for i, (chunk, score) in enumerate(zip(chunks, scores, strict=False), start=1):
        parts.append(
            f"[Source {i}] URL: {chunk.source_url}\nRelevance: {score:.3f}\n\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(
    query: str,
    chunks: list[Chunk],
    scores: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (system_blocks, messages) ready for the Anthropic API.

    Both the system prompt and the context block carry cache_control so that
    repeated queries hitting the same context pay only 0.1x the read cost.
    The question block intentionally has no cache_control — it changes every request.
    """
    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    context_text = _format_context(chunks, scores)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Context from Kubernetes documentation:\n\n{context_text}",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"Question: {query}",
                },
            ],
        }
    ]

    return system_blocks, messages
