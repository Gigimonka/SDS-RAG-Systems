from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm


DEFAULT_QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://127.0.0.1:6333",
)

DEFAULT_SOURCE_COLLECTION = os.getenv(
    "SOURCE_QDRANT_COLLECTION",
    "wikijs_docs",
)

DEFAULT_TARGET_COLLECTION = os.getenv(
    "TARGET_QDRANT_COLLECTION",
    "wikijs_docs_hybrid",
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

SPARSE_MODEL_NAME = os.getenv(
    "SPARSE_EMBEDDING_MODEL",
    "Qdrant/bm25",
)

SPARSE_LANGUAGE = os.getenv(
    "SPARSE_LANGUAGE",
    "russian",
)

BATCH_SIZE = int(
    os.getenv(
        "HYBRID_MIGRATION_BATCH_SIZE",
        "64",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Переносит существующие dense-векторы и payload "
            "в новую hybrid-коллекцию Qdrant."
        )
    )

    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
    )

    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE_COLLECTION,
        help="Исходная dense-коллекция.",
    )

    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_COLLECTION,
        help="Новая hybrid-коллекция.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Удалить целевую коллекцию, если она уже существует, "
            "и создать заново."
        ),
    )

    return parser.parse_args()


def make_sparse_model() -> SparseTextEmbedding:
    print(
        "Загрузка sparse-модели:",
        SPARSE_MODEL_NAME,
    )

    try:
        model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME,
            language=SPARSE_LANGUAGE,
        )

        print(
            "Язык BM25:",
            SPARSE_LANGUAGE,
        )

        return model

    except TypeError:
        # Совместимость со сборками FastEmbed,
        # где параметр language отсутствует.
        print(
            "Текущая версия FastEmbed не принимает "
            "параметр language; используется конфигурация "
            "модели по умолчанию."
        )

        return SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME,
        )


def extract_dense_vector(
    vector: Any,
) -> list[float]:
    """
    Старая коллекция может хранить:
      - один безымянный dense-вектор;
      - именованный dense-вектор.
    """

    if vector is None:
        raise ValueError(
            "У точки отсутствует dense-вектор."
        )

    if isinstance(
        vector,
        dict,
    ):
        if DENSE_VECTOR_NAME in vector:
            value = vector[
                DENSE_VECTOR_NAME
            ]

        elif len(vector) == 1:
            value = next(
                iter(
                    vector.values()
                )
            )

        else:
            available = ", ".join(
                map(
                    str,
                    vector.keys(),
                )
            )

            raise ValueError(
                "Не удалось выбрать dense-вектор. "
                f"Доступны: {available}"
            )

    else:
        value = vector

    if hasattr(
        value,
        "tolist",
    ):
        value = value.tolist()

    if not isinstance(
        value,
        list,
    ):
        value = list(
            value
        )

    return [
        float(item)
        for item in value
    ]


def make_sparse_text(
    payload: dict[str, Any],
) -> str:
    """
    В lexical-представление специально включаем
    заголовок и путь заголовков: это помогает
    точным названиям разделов, кодам, процедурам
    и идентификаторам.
    """

    parts = [
        str(
            payload.get(
                "title",
                "",
            )
        ).strip(),

        str(
            payload.get(
                "heading_path",
                "",
            )
        ).strip(),

        str(
            payload.get(
                "text",
                "",
            )
        ).strip(),
    ]

    return "\n\n".join(
        part
        for part in parts
        if part
    )


def sparse_to_qdrant(
    embedding: Any,
) -> models.SparseVector:
    indices = embedding.indices
    values = embedding.values

    if hasattr(
        indices,
        "tolist",
    ):
        indices = indices.tolist()

    if hasattr(
        values,
        "tolist",
    ):
        values = values.tolist()

    return models.SparseVector(
        indices=[
            int(item)
            for item in indices
        ],
        values=[
            float(item)
            for item in values
        ],
    )


def get_dense_size(
    client: QdrantClient,
    collection_name: str,
) -> int:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=False,
        with_vectors=True,
    )

    if not points:
        raise RuntimeError(
            "Исходная коллекция пуста."
        )

    return len(
        extract_dense_vector(
            points[0].vector
        )
    )


def create_target_collection(
    client: QdrantClient,
    target: str,
    dense_size: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(
        target
    )

    if exists and not recreate:
        raise RuntimeError(
            f"Коллекция '{target}' уже существует. "
            "Запусти с --recreate, если её нужно "
            "пересоздать."
        )

    if exists:
        print(
            "Удаление старой целевой коллекции:",
            target,
        )

        client.delete_collection(
            target
        )

    print(
        "Создание hybrid-коллекции:",
        target,
    )

    client.create_collection(
        collection_name=target,

        vectors_config={
            DENSE_VECTOR_NAME:
                models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                )
        },

        sparse_vectors_config={
            SPARSE_VECTOR_NAME:
                models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
        },
    )


def migrate(
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
    source: str,
    target: str,
    batch_size: int,
) -> tuple[int, int]:
    source_info = client.get_collection(
        source
    )

    expected_count = int(
        source_info.points_count
        or 0
    )

    offset = None
    migrated = 0
    skipped = 0

    progress = tqdm(
        total=expected_count,
        unit="точ.",
        desc="Hybrid-индексация",
    )

    while True:
        points, next_offset = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        texts = [
            make_sparse_text(
                point.payload
                or {}
            )
            for point in points
        ]

        sparse_embeddings = list(
            sparse_model.embed(
                texts,
                batch_size=min(
                    batch_size,
                    64,
                ),
            )
        )

        upload_points: list[
            models.PointStruct
        ] = []

        for (
            point,
            sparse_embedding,
        ) in zip(
            points,
            sparse_embeddings,
            strict=True,
        ):
            try:
                dense_vector = (
                    extract_dense_vector(
                        point.vector
                    )
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                skipped += 1

                print(
                    "\nПропущена точка",
                    point.id,
                    ":",
                    exc,
                    file=sys.stderr,
                )

                continue

            payload = dict(
                point.payload
                or {}
            )

            payload[
                "sparse_embedding_model"
            ] = SPARSE_MODEL_NAME

            payload[
                "hybrid_index"
            ] = True

            upload_points.append(
                models.PointStruct(
                    id=point.id,

                    vector={
                        DENSE_VECTOR_NAME:
                            dense_vector,

                        SPARSE_VECTOR_NAME:
                            sparse_to_qdrant(
                                sparse_embedding
                            ),
                    },

                    payload=payload,
                )
            )

        if upload_points:
            client.upsert(
                collection_name=target,
                points=upload_points,
                wait=True,
            )

            migrated += len(
                upload_points
            )

        progress.update(
            len(points)
        )

        if next_offset is None:
            break

        offset = next_offset

    progress.close()

    return (
        migrated,
        skipped,
    )


def main() -> None:
    args = parse_args()

    if args.source == args.target:
        raise RuntimeError(
            "Исходная и целевая коллекции "
            "должны отличаться."
        )

    client = QdrantClient(
        url=args.qdrant_url,
        timeout=120,
    )

    if not client.collection_exists(
        args.source
    ):
        raise RuntimeError(
            "Исходная коллекция "
            f"'{args.source}' не найдена."
        )

    dense_size = get_dense_size(
        client,
        args.source,
    )

    print(
        "Размер dense-вектора:",
        dense_size,
    )

    create_target_collection(
        client=client,
        target=args.target,
        dense_size=dense_size,
        recreate=args.recreate,
    )

    sparse_model = (
        make_sparse_model()
    )

    migrated, skipped = migrate(
        client=client,
        sparse_model=sparse_model,
        source=args.source,
        target=args.target,
        batch_size=args.batch_size,
    )

    target_info = client.get_collection(
        args.target
    )

    print()
    print(
        "Готово."
    )
    print(
        "Перенесено:",
        migrated,
    )
    print(
        "Пропущено:",
        skipped,
    )
    print(
        "Точек в новой коллекции:",
        target_info.points_count,
    )
    print()
    print(
        "Теперь укажи в окружении FastAPI:"
    )
    print(
        f"QDRANT_COLLECTION={args.target}"
    )


if __name__ == "__main__":
    main()