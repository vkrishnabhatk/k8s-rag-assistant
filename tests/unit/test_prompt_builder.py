from __future__ import annotations

from rag_assistant.generation.prompt_builder import SYSTEM_PROMPT, build_messages
from rag_assistant.ingestion.chunker import Chunk


def _chunk(i: int, text: str = "Kubernetes pods are the smallest deployable units.") -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        text=text,
        source_url=f"https://kubernetes.io/docs/page{i}/",
        source_title=f"Page {i}",
        word_count=len(text.split()),
        chunk_index=i,
    )


def test_system_block_has_cache_control():
    system_blocks, _ = build_messages("What is a pod?", [_chunk(0)], [0.9])
    assert len(system_blocks) == 1
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert system_blocks[0]["type"] == "text"


def test_system_block_contains_system_prompt():
    system_blocks, _ = build_messages("What is a pod?", [_chunk(0)], [0.9])
    assert SYSTEM_PROMPT in system_blocks[0]["text"]


def test_context_block_has_cache_control():
    _, messages = build_messages("What is a pod?", [_chunk(0)], [0.9])
    context_block = messages[0]["content"][0]
    assert context_block["cache_control"] == {"type": "ephemeral"}


def test_question_block_has_no_cache_control():
    _, messages = build_messages("What is a pod?", [_chunk(0)], [0.9])
    question_block = messages[0]["content"][1]
    assert "cache_control" not in question_block


def test_question_appears_in_question_block():
    question = "How do I scale a Deployment?"
    _, messages = build_messages(question, [_chunk(0)], [0.9])
    assert question in messages[0]["content"][1]["text"]


def test_source_url_appears_in_context():
    chunk = _chunk(0)
    _, messages = build_messages("What is a pod?", [chunk], [0.9])
    context_text = messages[0]["content"][0]["text"]
    assert chunk.source_url in context_text


def test_relevance_score_appears_in_context():
    _, messages = build_messages("What is a pod?", [_chunk(0)], [0.856])
    context_text = messages[0]["content"][0]["text"]
    assert "0.856" in context_text


def test_multiple_chunks_all_included():
    chunks = [_chunk(i) for i in range(3)]
    scores = [0.9, 0.8, 0.7]
    _, messages = build_messages("question?", chunks, scores)
    context_text = messages[0]["content"][0]["text"]
    for chunk in chunks:
        assert chunk.source_url in context_text


def test_messages_structure_is_valid():
    system_blocks, messages = build_messages("What is a pod?", [_chunk(0)], [0.9])
    assert isinstance(system_blocks, list)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert len(messages[0]["content"]) == 2
