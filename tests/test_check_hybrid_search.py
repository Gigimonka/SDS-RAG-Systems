from __future__ import annotations

import unittest

from tools.check_hybrid_search import (
    validate_health,
    validate_search_result,
)


class HybridSearchSmokeValidationTest(unittest.TestCase):
    def test_accepts_hybrid_rerank_health(self) -> None:
        rerank_enabled = validate_health(
            {
                "retrieval": "hybrid_rrf_rerank",
                "rerank_enabled": True,
                "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                "reranker_device": "cuda",
                "rerank_candidates": 40,
                "rerank_max_length": 768,
                "max_chunks_per_section": 2,
            }
        )

        self.assertTrue(rerank_enabled)

    def test_accepts_hybrid_without_reranker(self) -> None:
        self.assertFalse(
            validate_health(
                {
                    "retrieval": "hybrid_rrf",
                    "rerank_enabled": False,
                }
            )
        )

    def test_rejects_non_hybrid_mode(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_health(
                {
                    "retrieval": "dense",
                }
            )

    def test_reranked_result_requires_both_stage_scores(self) -> None:
        validate_search_result(
            {
                "score": 4.2,
                "retrieval_score": 0.03,
                "rerank_score": 4.2,
                "chunk_count": 2,
                "chunk_indices": [4, 5],
            },
            rerank_enabled=True,
        )

        with self.assertRaises(RuntimeError):
            validate_search_result(
                {
                    "score": 0.03,
                    "retrieval_score": 0.03,
                    "rerank_score": None,
                    "chunk_count": 1,
                    "chunk_indices": [4],
                },
                rerank_enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
