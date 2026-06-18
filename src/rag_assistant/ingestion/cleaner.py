from __future__ import annotations

import re
import unicodedata


def clean_text(raw: str) -> str:
    """Strip HTML artifacts, normalize whitespace, and remove Unicode noise."""
    # Normalize unicode to NFC (composed form)
    text = unicodedata.normalize("NFC", raw)

    # Remove zero-width and non-printable characters
    text = re.sub(r"[​‌‍﻿­]", "", text)

    # Collapse multiple spaces / tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are purely whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)

    return text.strip()
