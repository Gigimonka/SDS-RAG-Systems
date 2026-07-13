import os

import torch

from qdrant_client import (
    QdrantClient,
)

from sentence_transformers import (
    SentenceTransformer,
)


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://127.0.0.1:6333",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "wikijs_docs",
)

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)


def main() -> None:

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Загрузка embedding-модели..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    client = QdrantClient(
        url=QDRANT_URL,
    )

    while True:

        print()

        query = input(
            "Введите вопрос: "
        ).strip()

        if not query:

            continue

        if query.lower() in {
            "exit",
            "quit",
            "выход",
        }:

            break

        query_vector = model.encode(

            query,

            normalize_embeddings=True,

            convert_to_numpy=True,
        )

        results = client.query_points(

            collection_name=
                COLLECTION_NAME,

            query=
                query_vector.tolist(),

            limit=5,

            with_payload=True,

        ).points

        print(
            "\n"
            + "=" * 80
        )

        for number, result in enumerate(
            results,
            start=1,
        ):

            payload = (
                result.payload
                or {}
            )

            print(
                f"\nРЕЗУЛЬТАТ {number}"
            )

            print(
                f"Score: "
                f"{result.score:.4f}"
            )

            print(
                "Документ:",
                payload.get(
                    "title"
                ),
            )

            print(
                "Раздел:",
                payload.get(
                    "heading_path"
                ),
            )

            print(
                "Файл:",
                payload.get(
                    "source_path"
                ),
            )

            text = (
                payload.get(
                    "text",
                    "",
                )
            )

            print(
                "\nТекст:\n"
            )

            print(
                text
            )

            print(
                "\n"
                + "-" * 80
            )


if __name__ == "__main__":
    main()