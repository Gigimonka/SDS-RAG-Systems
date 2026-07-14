from __future__ import annotations

import unittest

from sds_rag.indexing.build import (
    count_tokens,
    pack_blocks,
    split_long_text,
)


class CharacterTokenizer:
    """Простой токенизатор для проверки алгоритма без загрузки модели."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        verbose: bool,
    ) -> list[int]:
        tokens = [ord(character) for character in text]
        return [0, *tokens, 1] if add_special_tokens else tokens

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id > 1)


class IndexingChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def test_long_single_sentence_is_split_by_hard_token_limit(self) -> None:
        parts = split_long_text(
            "а" * 25,
            self.tokenizer,
            max_tokens=10,
        )

        self.assertEqual([len(part) for part in parts], [10, 10, 5])
        self.assertTrue(
            all(count_tokens(part, self.tokenizer) <= 10 for part in parts)
        )

    def test_pack_blocks_keeps_overlap_inside_token_limit(self) -> None:
        chunks = pack_blocks(
            ["а" * 8, "б" * 8, "в" * 8],
            self.tokenizer,
            max_tokens=12,
            overlap_tokens=3,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(count_tokens(chunk, self.tokenizer) <= 12 for chunk in chunks)
        )
        self.assertTrue(chunks[1].startswith("а"))
        self.assertLessEqual(chunks[1].count("а"), 3)

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pack_blocks(
                ["текст"],
                self.tokenizer,
                max_tokens=10,
                overlap_tokens=10,
            )


if __name__ == "__main__":
    unittest.main()
