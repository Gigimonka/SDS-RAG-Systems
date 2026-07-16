from __future__ import annotations

import unittest
from types import SimpleNamespace

from sds_rag.core.reranking import (
    RankedPoint,
    group_ranked_sections,
    make_rerank_document,
    merge_ranked_section_text,
    merge_unique_points,
    rerank_points,
    select_identifier_points,
)


def make_point(
    point_id: str,
    score: float,
    *,
    text: str,
    source_path: str = "document.md",
    heading_path: str = "Раздел",
    chunk_index: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "title": "Документ",
            "heading_path": heading_path,
            "text": text,
            "source_path": source_path,
            "chunk_index": chunk_index,
        },
    )


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.inputs: list[tuple[str, str]] = []
        self.kwargs: dict = {}

    def predict(self, inputs, **kwargs):
        self.inputs = list(inputs)
        self.kwargs = kwargs
        return self.scores


class RerankingTest(unittest.TestCase):
    def test_document_contains_only_relevant_payload_fields(self) -> None:
        document = make_rerank_document(
            {
                "title": "Настройка",
                "heading_path": "Интеграция → ASTM",
                "text": "Укажите обработчик.",
                "absolute_path": "/private/path.md",
            }
        )

        self.assertIn("Документ: Настройка", document)
        self.assertIn("Раздел: Интеграция → ASTM", document)
        self.assertIn("Укажите обработчик.", document)
        self.assertNotIn("/private/path.md", document)

    def test_cross_encoder_score_reorders_rrf_candidates(self) -> None:
        first = make_point("first", 0.9, text="Общий текст")
        second = make_point("second", 0.5, text="Точный ответ")
        reranker = FakeReranker([-2.0, 6.5])

        ranked = rerank_points(
            "Где точный ответ?",
            [first, second],
            reranker,
            batch_size=4,
        )

        self.assertEqual([item.point.id for item in ranked], ["second", "first"])
        self.assertEqual(ranked[0].retrieval_score, 0.5)
        self.assertEqual(ranked[0].rerank_score, 6.5)
        self.assertEqual(reranker.inputs[1][0], "Где точный ответ?")
        self.assertIn("Точный ответ", reranker.inputs[1][1])
        self.assertEqual(reranker.kwargs["batch_size"], 4)
        self.assertFalse(reranker.kwargs["show_progress_bar"])

    def test_multiple_identifiers_may_come_from_different_chunks(self) -> None:
        first = make_point("a", 0.9, text="Параметр CODE_ALPHA_1")
        second = make_point("b", 0.8, text="Параметр CODE_BETA_2")
        unrelated = make_point("other", 0.7, text="Другой документ")

        selected = select_identifier_points(
            [first, unrelated, second],
            ["CODE_ALPHA_1", "CODE_BETA_2"],
            limit=2,
        )

        self.assertEqual([point.id for point in selected], ["a", "b"])

    def test_identifier_selection_is_empty_when_any_code_is_missing(self) -> None:
        point = make_point("a", 0.9, text="Параметр CODE_ALPHA_1")

        selected = select_identifier_points(
            [point],
            ["CODE_ALPHA_1", "CODE_MISSING_2"],
            limit=10,
        )

        self.assertEqual(selected, [])

    def test_merge_keeps_first_point_instance(self) -> None:
        first = make_point("same", 0.9, text="Первый")
        duplicate = make_point("same", 0.1, text="Дубликат")
        second = make_point("second", 0.8, text="Второй")

        merged = merge_unique_points([first], [duplicate, second])

        self.assertEqual(merged, [first, second])

    def test_section_group_keeps_two_chunks_after_reranking(self) -> None:
        weaker_chunk = make_point(
            "weak",
            0.9,
            text="Слабый чанк",
            chunk_index=10,
        )
        better_chunk = make_point(
            "best",
            0.7,
            text="Лучший чанк",
            chunk_index=11,
        )
        third_chunk = make_point(
            "third",
            0.6,
            text="Третий чанк",
            chunk_index=12,
        )
        other_section = make_point(
            "other",
            0.5,
            text="Другой раздел",
            heading_path="Другой раздел",
        )
        ranked = [
            RankedPoint(better_chunk, 0.7, 7.0),
            RankedPoint(weaker_chunk, 0.9, 1.0),
            RankedPoint(third_chunk, 0.6, 0.8),
            RankedPoint(other_section, 0.5, 0.5),
        ]

        sections = group_ranked_sections(
            ranked,
            limit=10,
            max_chunks_per_section=2,
        )

        self.assertEqual(len(sections), 2)
        self.assertEqual(
            [item.point.id for item in sections[0].ranked_points],
            ["best", "weak"],
        )
        self.assertEqual(sections[0].primary.point.id, "best")

    def test_section_text_is_ordered_and_exact_overlap_is_removed(self) -> None:
        overlap = "повторяющийся фрагмент длиной больше тридцати символов"
        later = make_point(
            "later",
            0.9,
            text=f"{overlap} и продолжение",
            chunk_index=6,
        )
        earlier = make_point(
            "earlier",
            0.8,
            text=f"Начало и {overlap}",
            chunk_index=5,
        )
        section = group_ranked_sections(
            [
                RankedPoint(later, 0.9, 9.0),
                RankedPoint(earlier, 0.8, 8.0),
            ],
            limit=1,
            max_chunks_per_section=2,
        )[0]

        merged = merge_ranked_section_text(section)

        self.assertTrue(merged.startswith("Начало"))
        self.assertEqual(merged.count(overlap), 1)
        self.assertTrue(merged.endswith("и продолжение"))

    def test_non_adjacent_chunks_get_a_visible_separator(self) -> None:
        first = make_point("first", 0.9, text="Начало", chunk_index=2)
        third = make_point("third", 0.8, text="Конец", chunk_index=4)
        section = group_ranked_sections(
            [
                RankedPoint(first, 0.9, 9.0),
                RankedPoint(third, 0.8, 8.0),
            ],
            limit=1,
            max_chunks_per_section=2,
        )[0]

        merged = merge_ranked_section_text(section)

        self.assertIn("другой релевантный фрагмент", merged)


if __name__ == "__main__":
    unittest.main()
