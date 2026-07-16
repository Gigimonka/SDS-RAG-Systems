"""Pure ranking helpers for the hybrid retrieval pipeline.

The module deliberately does not load a model.  FastAPI owns the expensive
CrossEncoder instance, while these helpers remain cheap to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .helpers import payload_contains_identifier


class Reranker(Protocol):
    """The part of ``sentence_transformers.CrossEncoder`` used by the API."""

    def predict(
        self,
        inputs: list[tuple[str, str]],
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class RankedPoint:
    """A Qdrant point together with scores from each retrieval stage."""

    point: Any
    retrieval_score: float
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """Return the score that defines the final ordering."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.retrieval_score


@dataclass(frozen=True, slots=True)
class RankedSection:
    """One ranked source section containing one or more relevant chunks."""

    section_key: tuple[str, str]
    ranked_points: tuple[RankedPoint, ...]

    @property
    def primary(self) -> RankedPoint:
        """Return the chunk that determined the section's rank."""
        return self.ranked_points[0]

    @property
    def score(self) -> float:
        return self.primary.score

    @property
    def retrieval_score(self) -> float:
        return self.primary.retrieval_score

    @property
    def rerank_score(self) -> float | None:
        return self.primary.rerank_score


def make_rerank_document(payload: dict[str, Any]) -> str:
    """Build the passage shown to the cross-encoder."""
    title = str(payload.get("title", "")).strip()
    heading_path = str(payload.get("heading_path", "")).strip()
    text = str(payload.get("text", "")).strip()

    return (
        f"Документ: {title}\n"
        f"Раздел: {heading_path}\n\n"
        f"{text}"
    ).strip()


def rank_by_retrieval_score(points: Sequence[Any]) -> list[RankedPoint]:
    """Wrap Qdrant results without changing their ranking stage."""
    return [
        RankedPoint(
            point=point,
            retrieval_score=float(point.score),
        )
        for point in points
    ]


def _score_to_float(value: Any) -> float:
    """Normalize numpy, torch and one-item sequence score representations."""
    if hasattr(value, "item"):
        return float(value.item())

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Ожидалась одна оценка реранкера на документ")
        return _score_to_float(value[0])

    return float(value)


def rerank_points(
    question: str,
    points: Sequence[Any],
    reranker: Reranker,
    *,
    batch_size: int,
) -> list[RankedPoint]:
    """Score query-passage pairs and return points in cross-encoder order."""
    if not points:
        return []

    pairs = [
        (
            question,
            make_rerank_document(point.payload or {}),
        )
        for point in points
    ]

    raw_scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    scores = [_score_to_float(score) for score in raw_scores]

    if len(scores) != len(points):
        raise RuntimeError(
            "Реранкер вернул неожиданное количество оценок: "
            f"{len(scores)} вместо {len(points)}"
        )

    ranked = [
        RankedPoint(
            point=point,
            retrieval_score=float(point.score),
            rerank_score=score,
        )
        for point, score in zip(
            points,
            scores,
            strict=True,
        )
    ]

    return sorted(
        ranked,
        key=lambda item: (
            item.score,
            item.retrieval_score,
        ),
        reverse=True,
    )


def merge_unique_points(*groups: Sequence[Any]) -> list[Any]:
    """Merge Qdrant result sets while preserving the first occurrence."""
    merged: list[Any] = []
    seen_ids: set[str] = set()

    for group in groups:
        for point in group:
            point_id = str(point.id)

            if point_id in seen_ids:
                continue

            seen_ids.add(point_id)
            merged.append(point)

    return merged


def select_identifier_points(
    points: Sequence[Any],
    identifiers: Sequence[str],
    *,
    limit: int,
) -> list[Any]:
    """Keep candidates mentioning any requested identifier.

    Every identifier must be found somewhere, but identifiers do not have to
    occur in the same chunk.  One candidate per identifier is protected before
    the rest of the RRF-ordered pool is filled.
    """
    normalized = list(dict.fromkeys(item.upper() for item in identifiers))

    if not normalized:
        return list(points[:limit])

    matches: list[tuple[Any, frozenset[str]]] = []
    found: set[str] = set()

    for point in points:
        payload = point.payload or {}
        point_matches = frozenset(
            identifier
            for identifier in normalized
            if payload_contains_identifier(
                payload,
                identifier,
            )
        )

        if not point_matches:
            continue

        found.update(point_matches)
        matches.append((point, point_matches))

    if found != set(normalized):
        return []

    protected_ids: set[str] = set()

    for identifier in normalized:
        for point, point_matches in matches:
            if identifier in point_matches:
                protected_ids.add(str(point.id))
                break

    selected = [
        point
        for point, _ in matches
        if str(point.id) in protected_ids
    ]
    selected_ids = {str(point.id) for point in selected}

    for point, _ in matches:
        if len(selected) >= limit:
            break

        point_id = str(point.id)

        if point_id in selected_ids:
            continue

        selected_ids.add(point_id)
        selected.append(point)

    return selected


def _section_key(ranked_point: RankedPoint) -> tuple[str, str]:
    payload = ranked_point.point.payload or {}
    return (
        str(payload.get("source_path", "")),
        str(payload.get("heading_path", "")),
    )


def group_ranked_sections(
    ranked_points: Sequence[RankedPoint],
    *,
    limit: int,
    max_chunks_per_section: int,
) -> list[RankedSection]:
    """Keep the best sections and up to N reranked chunks inside each one."""
    if limit <= 0:
        raise ValueError("limit должен быть положительным")

    if max_chunks_per_section <= 0:
        raise ValueError("max_chunks_per_section должен быть положительным")

    section_order: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[RankedPoint]] = {}

    for ranked_point in ranked_points:
        section_key = _section_key(ranked_point)
        section_points = grouped.get(section_key)

        if section_points is None:
            if len(section_order) >= limit:
                continue

            section_order.append(section_key)
            grouped[section_key] = [ranked_point]
            continue

        if len(section_points) < max_chunks_per_section:
            section_points.append(ranked_point)

    return [
        RankedSection(
            section_key=section_key,
            ranked_points=tuple(grouped[section_key]),
        )
        for section_key in section_order
    ]


def _chunk_index(ranked_point: RankedPoint) -> int | None:
    value = (ranked_point.point.payload or {}).get("chunk_index")

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def section_points_in_document_order(
    section: RankedSection,
) -> list[RankedPoint]:
    """Order selected chunks by their location while preserving unknowns."""
    indexed_points = [
        (
            position,
            ranked_point,
            _chunk_index(ranked_point),
        )
        for position, ranked_point in enumerate(section.ranked_points)
    ]
    indexed_points.sort(
        key=lambda item: (
            item[2] is None,
            item[2] if item[2] is not None else item[0],
        )
    )
    return [ranked_point for _, ranked_point, _ in indexed_points]


def _remove_exact_overlap(
    previous_text: str,
    current_text: str,
    *,
    min_overlap_chars: int = 32,
) -> str:
    """Remove the exact overlap produced by adjacent indexing chunks."""
    max_overlap = min(
        len(previous_text),
        len(current_text),
        2_048,
    )

    for size in range(max_overlap, min_overlap_chars - 1, -1):
        if previous_text[-size:] == current_text[:size]:
            return current_text[size:].lstrip()

    return current_text


def merge_ranked_section_text(
    section: RankedSection,
) -> str:
    """Merge selected section chunks into one citation-friendly context."""
    ordered_points = section_points_in_document_order(section)

    if not ordered_points:
        return ""

    first_payload = ordered_points[0].point.payload or {}
    previous_text = str(first_payload.get("text", "")).strip()
    previous_index = _chunk_index(ordered_points[0])
    parts = [previous_text] if previous_text else []

    for ranked_point in ordered_points[1:]:
        payload = ranked_point.point.payload or {}
        current_text = str(payload.get("text", "")).strip()
        current_index = _chunk_index(ranked_point)

        if not current_text:
            continue

        adjacent = (
            previous_index is not None
            and current_index is not None
            and current_index == previous_index + 1
        )

        if adjacent:
            current_text = _remove_exact_overlap(
                previous_text,
                current_text,
            )
        else:
            current_text = (
                "[... другой релевантный фрагмент "
                "этого же раздела ...]\n\n"
                + current_text
            )

        if current_text:
            parts.append(current_text)

        previous_text = str(payload.get("text", "")).strip()
        previous_index = current_index

    return "\n\n".join(parts)
