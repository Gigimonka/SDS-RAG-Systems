"""Smoke-test the running hybrid-search API."""

from __future__ import annotations

import json
import os

import httpx


API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)

QUESTIONS = [
    "Как сделать связки с ЕРЛУ?",
    "LP__HELIX_COMPARE_1",
    "CF_LABCOUNTERLINKS_SEARCH",
    "AN_HANDLER_BS240_ASTM",
]


def validate_health(health_data: dict) -> bool:
    """Validate retrieval capabilities and return whether reranking is active."""
    retrieval_mode = str(
        health_data.get(
            "retrieval",
            "",
        )
    )

    if not retrieval_mode.startswith("hybrid_rrf"):
        raise RuntimeError(
            "API не использует hybrid RRF: "
            f"retrieval={retrieval_mode!r}."
        )

    rerank_enabled = health_data.get("rerank_enabled") is True

    if rerank_enabled:
        if "rerank" not in retrieval_mode:
            raise RuntimeError(
                "Health сообщает о включённом реранкере, "
                "но retrieval mode не содержит rerank."
            )

        for field in (
            "reranker_model",
            "reranker_device",
            "rerank_candidates",
            "rerank_max_length",
            "max_chunks_per_section",
        ):
            if health_data.get(field) in {None, ""}:
                raise RuntimeError(
                    "Health не содержит настройку реранкера: "
                    f"{field}."
                )

    return rerank_enabled


def validate_search_result(
    item: dict,
    *,
    rerank_enabled: bool,
) -> None:
    """Ensure the search response exposes scores for every active stage."""
    if not isinstance(item.get("score"), (int, float)):
        raise RuntimeError("Результат поиска не содержит итоговый score.")

    if not isinstance(item.get("retrieval_score"), (int, float)):
        raise RuntimeError("Результат поиска не содержит retrieval_score.")

    if rerank_enabled and not isinstance(
        item.get("rerank_score"),
        (int, float),
    ):
        raise RuntimeError(
            "Реранкер включён, но результат не содержит rerank_score. "
            "Проверьте лог API: возможно, сработал RRF fallback."
        )

    chunk_count = item.get("chunk_count")
    chunk_indices = item.get("chunk_indices")

    if not isinstance(chunk_count, int) or chunk_count <= 0:
        raise RuntimeError("Результат поиска не содержит корректный chunk_count.")

    if not isinstance(chunk_indices, list) or not all(
        isinstance(index, int) for index in chunk_indices
    ):
        raise RuntimeError("Результат поиска не содержит chunk_indices.")


def main() -> None:
    with httpx.Client(
        timeout=120,
    ) as client:
        health = client.get(
            f"{API_URL}/health"
        )

        health.raise_for_status()

        health_data = health.json()

        print(
            json.dumps(
                health_data,
                ensure_ascii=False,
                indent=2,
            )
        )

        rerank_enabled = validate_health(health_data)

        print(
            "\nHybrid search активен"
            + (
                " вместе с reranker.\n"
                if rerank_enabled
                else ".\n"
            )
        )

        for question in QUESTIONS:
            response = client.post(
                f"{API_URL}/search",

                json={
                    "question":
                        question,

                    "limit":
                        5,
                },
            )

            response.raise_for_status()

            data = response.json()

            print(
                "=" * 80
            )

            print(
                "ВОПРОС:",
                question,
            )

            for index, item in enumerate(
                data.get(
                    "results",
                    [],
                ),
                start=1,
            ):
                validate_search_result(
                    item,
                    rerank_enabled=rerank_enabled,
                )

                print(
                    f"\n{index}. "
                    f"score={item['score']} "
                    f"rrf={item['retrieval_score']} "
                    f"rerank={item.get('rerank_score')} "
                    f"chunks={item.get('chunk_indices')}"
                )

                print(
                    item.get(
                        "title",
                        "",
                    )
                )

                print(
                    item.get(
                        "heading_path",
                        "",
                    )
                )

                print(
                    item.get(
                        "wiki_url",
                        "",
                    )
                )


if __name__ == "__main__":
    main()
