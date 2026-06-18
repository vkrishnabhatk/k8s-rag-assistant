from __future__ import annotations

from collections.abc import Generator
from typing import Any

import anthropic
import structlog

from rag_assistant.config import Settings

log = structlog.get_logger(__name__)

_CACHE_BETA_HEADER = {"anthropic-beta": "prompt-caching-2024-07-31"}


class LLMClient:
    """Thin wrapper around the Anthropic SDK supporting blocking and streaming calls."""

    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
        self._model = settings.anthropic_model
        self._max_tokens = settings.anthropic_max_tokens

    def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Blocking completion with prompt caching enabled.

        Returns (answer_text, usage_dict).
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            extra_headers=_CACHE_BETA_HEADER,
        )
        first_block = response.content[0]
        answer = first_block.text if isinstance(first_block, anthropic.types.TextBlock) else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        }
        log.debug(
            "llm_complete",
            model=self._model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read=usage["cache_read_input_tokens"],
        )
        return answer, usage

    def stream(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Yield text delta strings as they arrive from the streaming API."""
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            extra_headers=_CACHE_BETA_HEADER,
        ) as stream:
            yield from stream.text_stream
