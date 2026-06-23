from __future__ import annotations

from typing import Any

from rag_assistant.ingestion.chunker import Chunk

SYSTEM_PROMPT = """\
You are a Kubernetes assistant. Answer every question using exactly one of two modes:

DOCUMENTATION MODE (preferred): When the provided context contains relevant information,
answer using that context and cite the source URL(s). Do not add any prefix.

GENERAL KNOWLEDGE MODE: When the context lacks sufficient information, use your training
knowledge to answer. This applies to ALL of these topics even if the context does not
define them: node pools, node groups, CNI plugins, kubectl, Helm, Operators, CRDs,
kube-proxy, kubelet, kubeadm, etcd, cloud-managed K8s (GKE/EKS/AKS), service meshes,
container runtimes, and any other Kubernetes or cloud-native concept.
Start your response with exactly "Based on general Kubernetes knowledge:" — nothing before it.

OUT OF SCOPE: Only use this when the question has absolutely nothing to do with
Kubernetes, containers, or cloud infrastructure (e.g. cooking, sports, history).
Respond with exactly: "This question is outside the scope of Kubernetes documentation."

Rules:
- Pick exactly one mode. Never mix them.
- Never say "I don't have enough information" — always use general knowledge mode for K8s topics.
- Be concise and technically precise.\
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
