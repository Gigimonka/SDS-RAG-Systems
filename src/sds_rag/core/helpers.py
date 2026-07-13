"""Pure helpers shared by the RAG API and maintenance tools.

This module intentionally has no dependency on FastAPI application state,
embedding models, or external services.  Keeping text-processing rules here
makes them cheap to test and prevents the API module from becoming a second
utility library.
"""

from __future__ import annotations

import re
from typing import Any

from qdrant_client import models


_CITATION_BLOCK_RE = re.compile(
    r"\[(?:Источник|Источники)\s+([0-9,\s]+)\]",
    flags=re.IGNORECASE,
)
_TECHNICAL_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]*)+\b"
)
_NO_ANSWER_MARKERS = (
    "в документации не найдено",
    "в документации недостаточно информации",
    "в предоставленных источниках не найдено",
    "в источниках не найдено",
    "не упоминается в предоставленных источниках",
    "невозможно дать точный ответ",
    "недостаточно информации для однозначного ответа",
)
_EXACT_FOLLOW_UPS = frozenset(
    {
        "че там",
        "чё там",
        "что там",
        "ну что",
        "ну че",
        "ну чё",
        "что дальше",
        "дальше",
        "продолжай",
        "подробнее",
        "можно подробнее",
        "почему",
        "где",
        "как",
        "и что",
        "и что дальше",
    }
)
_CONTINUATION_MARKERS = (
    "а ",
    "и ",
    "но ",
    "тогда ",
    "это ",
    "этот ",
    "эта ",
    "эти ",
    "там ",
    "тут ",
    "как это",
    "где это",
    "почему так",
    "что с этим",
    "а если",
    "а как",
    "а где",
    "а почему",
    "можешь подробнее",
    "расскажи подробнее",
)
_SEARCHABLE_PAYLOAD_FIELDS = (
    "title",
    "heading_path",
    "text",
    "source_path",
    "absolute_path",
)


def content_to_text(content: Any) -> str:
    """Normalize string and block-based OpenAI message content."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if value:
                    parts.append(str(value))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if content is None:
        return ""

    return str(content).strip()


def clean_history_message(text: str) -> str:
    """Remove the generated source list before reusing an old answer."""
    return re.split(
        r"\n-{3,}\s*\n#{1,6}\s*Источники\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def extract_cited_source_numbers(answer: str) -> list[int]:
    """Return unique source numbers in ascending order."""
    found = {
        int(value)
        for block in _CITATION_BLOCK_RE.findall(answer)
        for value in re.findall(r"\d+", block)
    }
    return sorted(found)


def is_no_answer(answer: str) -> bool:
    """Detect an explicit statement that documentation is insufficient."""
    normalized = answer.strip().lower()
    return any(marker in normalized for marker in _NO_ANSWER_MARKERS)


def is_follow_up_question(question: str) -> bool:
    """Detect short questions that depend on previous conversation context."""
    normalized = question.strip().lower().rstrip("?!.,:; ")
    if len(normalized.split()) > 12:
        return False
    return normalized in _EXACT_FOLLOW_UPS or normalized.startswith(
        _CONTINUATION_MARKERS
    )


def extract_technical_identifiers(question: str) -> list[str]:
    """Extract stable, unique identifiers such as ``LP__HELIX_COMPARE_1``."""
    candidates = _TECHNICAL_IDENTIFIER_RE.findall(question)
    return list(
        dict.fromkeys(
            candidate.upper() for candidate in candidates if len(candidate) >= 5
        )
    )


def payload_contains_identifier(
    payload: dict[str, Any],
    identifier: str,
) -> bool:
    """Check all indexed textual payload fields for an exact identifier."""
    searchable_text = "\n".join(
        str(payload.get(field, "")) for field in _SEARCHABLE_PAYLOAD_FIELDS
    ).upper()
    return identifier.upper() in searchable_text


def sparse_embedding_to_vector(embedding: Any) -> models.SparseVector:
    """Convert a FastEmbed sparse result into Qdrant's transport model."""
    indices = embedding.indices
    values = embedding.values

    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()

    return models.SparseVector(
        indices=[int(item) for item in indices],
        values=[float(item) for item in values],
    )


def split_stream_text(text: str, chunk_size: int = 48) -> list[str]:
    """Split a completed response into deterministic SSE payload chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
