from __future__ import annotations

import unittest
from types import SimpleNamespace

from sds_rag.core.helpers import (
    clean_history_message,
    content_to_text,
    extract_cited_source_numbers,
    extract_technical_identifiers,
    is_follow_up_question,
    is_no_answer,
    payload_contains_identifier,
    sparse_embedding_to_vector,
    split_stream_text,
)


class RagHelpersTest(unittest.TestCase):
    def test_content_to_text_supports_openai_content_blocks(self) -> None:
        content = [
            {"type": "text", "text": "Первый блок"},
            {"content": "Второй блок"},
            None,
        ]

        self.assertEqual(content_to_text(content), "Первый блок\nВторой блок")
        self.assertEqual(content_to_text(None), "")

    def test_clean_history_message_removes_generated_sources(self) -> None:
        answer = "Полезный ответ.\n\n---\n### Источники\n- [Документ](url)"

        self.assertEqual(clean_history_message(answer), "Полезный ответ.")

    def test_citation_numbers_are_unique_and_sorted(self) -> None:
        answer = "[Источник 3] [Источники 2, 3] [Источник 1]"

        self.assertEqual(extract_cited_source_numbers(answer), [1, 2, 3])

    def test_no_answer_detection(self) -> None:
        self.assertTrue(is_no_answer("В документации недостаточно информации."))
        self.assertFalse(is_no_answer("Ответ подтверждён. [Источник 1]"))

    def test_follow_up_detection(self) -> None:
        self.assertTrue(is_follow_up_question("А где это?"))
        self.assertTrue(is_follow_up_question("Можно подробнее?"))
        self.assertFalse(
            is_follow_up_question(
                "Как настроить интеграцию лабораторного анализатора по ASTM?"
            )
        )

    def test_technical_identifiers_are_normalized_and_deduplicated(self) -> None:
        question = "Сравни lp__helix_compare_1 и LP__HELIX_COMPARE_1 с A_B_C"

        self.assertEqual(
            extract_technical_identifiers(question),
            ["LP__HELIX_COMPARE_1", "A_B_C"],
        )

    def test_payload_identifier_search_covers_metadata_and_text(self) -> None:
        payload = {
            "title": "Настройка анализатора",
            "text": "Используется AN_HANDLER_BS240_ASTM.",
        }

        self.assertTrue(
            payload_contains_identifier(payload, "an_handler_bs240_astm")
        )
        self.assertFalse(payload_contains_identifier(payload, "UNKNOWN_CODE"))

    def test_sparse_embedding_conversion_normalizes_number_types(self) -> None:
        embedding = SimpleNamespace(indices=(1, 4), values=(0.25, 2))

        vector = sparse_embedding_to_vector(embedding)

        self.assertEqual(vector.indices, [1, 4])
        self.assertEqual(vector.values, [0.25, 2.0])

    def test_stream_split_rejects_invalid_chunk_size(self) -> None:
        self.assertEqual(split_stream_text("abcdef", 2), ["ab", "cd", "ef"])
        with self.assertRaises(ValueError):
            split_stream_text("abcdef", 0)


if __name__ == "__main__":
    unittest.main()
