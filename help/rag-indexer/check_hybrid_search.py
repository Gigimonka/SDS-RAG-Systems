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

        if (
            health_data.get(
                "retrieval"
            )
            != "hybrid_rrf"
        ):
            raise RuntimeError(
                "API не использует hybrid_rrf."
            )

        print(
            "\nHybrid search активен.\n"
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
                print(
                    f"\n{index}. "
                    f"score={item['score']}"
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