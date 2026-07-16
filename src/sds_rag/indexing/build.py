"""Build a Qdrant index from exported Wiki.js Markdown documents."""

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Iterator, Protocol

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================
# НАСТРОЙКИ
# ============================================================

MD_ROOT = Path(os.getenv("MD_ROOT", "data/wikijs_export")).resolve()

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
    "ai-forever/FRIDA",
)


# Целевой размер содержимого чанка в токенах embedding-модели.
# Остаток окна модели резервируется под название документа,
# путь раздела и специальные токены.
MAX_CHUNK_TOKENS = 288

# Размер перекрытия между соседними чанками в токенах.
OVERLAP_TOKENS = 48

# Дополнительный запас на границе служебного текста и чанка.
TOKEN_SAFETY_MARGIN = 8

# Начни с 8.
# На RTX 5070 Ti потом можно попробовать 16 или 32.
EMBEDDING_BATCH_SIZE = 32

# Сколько точек отправлять в Qdrant одним запросом.
QDRANT_BATCH_SIZE = 64


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class TextTokenizer(Protocol):
    """Минимальный интерфейс токенизатора для разбиения документов."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        verbose: bool,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


def remove_frontmatter(text: str) -> str:
    """
    Удаляет YAML front matter:

    ---
    title: Страница
    description: ...
    ---
    """

    if not text.startswith("---"):
        return text

    lines = text.splitlines()

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :])

    return text


def clean_heading(text: str) -> str:
    """
    Удаляет лишние Markdown-символы из заголовка.
    """

    text = re.sub(
        r"\s+#+\s*$",
        "",
        text,
    )

    text = re.sub(
        r"\[([^\]]+)]\([^)]+\)",
        r"\1",
        text,
    )

    return text.strip()


def split_markdown_blocks(
    text: str,
) -> list[str]:
    """
    Делит содержимое раздела по пустым строкам.

    При этом fenced code block:

    ```sql
    SELECT ...
    ```

    остаётся одним блоком.
    """

    blocks: list[str] = []

    current: list[str] = []

    in_code = False

    fence_marker: str | None = None

    for line in text.splitlines():

        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]

            if not in_code:
                in_code = True
                fence_marker = marker

            elif marker == fence_marker:
                in_code = False
                fence_marker = None

            current.append(line)

            continue

        if not stripped and not in_code:

            if current:

                block = "\n".join(current).strip()

                if block:
                    blocks.append(block)

                current = []

            continue

        current.append(line)

    if current:

        block = "\n".join(current).strip()

        if block:
            blocks.append(block)

    return blocks


def count_tokens(
    text: str,
    tokenizer: TextTokenizer,
    *,
    add_special_tokens: bool = False,
) -> int:
    """Считает токены тем же токенизатором, который использует bi-encoder."""

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            verbose=False,
        )
    )


def split_by_token_limit(
    text: str,
    tokenizer: TextTokenizer,
    max_tokens: int,
) -> list[str]:
    """Жёстко делит текст, если одно предложение не помещается целиком."""

    if max_tokens <= 0:
        raise ValueError("max_tokens должен быть положительным")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )

    parts: list[str] = []

    for start in range(0, len(token_ids), max_tokens):
        part = tokenizer.decode(
            token_ids[start : start + max_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if part:
            parts.append(part)

    return parts


def split_long_text(
    text: str,
    tokenizer: TextTokenizer,
    max_tokens: int,
) -> list[str]:
    """
    Запасной вариант для очень длинного абзаца.

    Старается делить по предложениям.
    """

    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    parts: list[str] = []

    current: list[str] = []

    for sentence in sentences:

        sentence_tokens = count_tokens(sentence, tokenizer)

        if sentence_tokens > max_tokens:

            if current:
                parts.append(" ".join(current).strip())
                current = []

            parts.extend(
                split_by_token_limit(
                    sentence,
                    tokenizer,
                    max_tokens,
                )
            )

            continue

        candidate = " ".join(current + [sentence])

        if current and count_tokens(candidate, tokenizer) > max_tokens:

            parts.append(" ".join(current).strip())

            current = []

        current.append(sentence)

    if current:

        parts.append(" ".join(current).strip())

    return [part for part in parts if part]


def pack_blocks(
    blocks: list[str],
    tokenizer: TextTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Объединяет маленькие Markdown-блоки
    в чанки до max_tokens с ограниченным токенным перекрытием.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens должен быть положительным")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens не может быть отрицательным")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens должен быть меньше max_tokens")

    prepared_blocks: list[str] = []

    for block in blocks:

        prepared_blocks.extend(
            split_long_text(
                block,
                tokenizer,
                max_tokens,
            )
        )

    chunks: list[str] = []

    current = ""

    for block in prepared_blocks:

        candidate = f"{current}\n\n{block}".strip() if current else block

        if current and count_tokens(candidate, tokenizer) > max_tokens:

            chunks.append(current)

            if overlap_tokens > 0:
                current_ids = tokenizer.encode(
                    current,
                    add_special_tokens=False,
                    verbose=False,
                )
                block_tokens = count_tokens(block, tokenizer)
                overlap_budget = min(
                    overlap_tokens,
                    max(0, max_tokens - block_tokens),
                )
                overlap = tokenizer.decode(
                    current_ids[-overlap_budget:] if overlap_budget else [],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            else:
                overlap = ""

            candidate = f"{overlap}\n\n{block}".strip() if overlap else block

            while overlap and count_tokens(candidate, tokenizer) > max_tokens:
                overlap_ids = tokenizer.encode(
                    overlap,
                    add_special_tokens=False,
                    verbose=False,
                )
                overflow = count_tokens(candidate, tokenizer) - max_tokens
                overlap_ids = overlap_ids[max(1, overflow) :]
                overlap = tokenizer.decode(
                    overlap_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                candidate = f"{overlap}\n\n{block}".strip() if overlap else block

        current = candidate

    if current:
        chunks.append(current)

    return chunks


def first_h1(
    text: str,
) -> str | None:
    """
    Берёт первый заголовок первого уровня.
    """

    for line in text.splitlines():

        match = HEADING_RE.match(line)

        if not match:
            continue

        level = len(match.group(1))

        if level == 1:

            return clean_heading(match.group(2))

    return None


def split_into_sections(
    text: str,
    tokenizer: TextTokenizer,
) -> list[dict]:
    """
    Делит Markdown по заголовкам.

    Для каждого блока сохраняет:

    Лабораторный портал
      → Настройка
        → HTTPS
    """

    heading_stack: list[str] = []

    sections: list[dict] = []

    current_lines: list[str] = []

    current_heading_path = ""

    def flush_section() -> None:

        nonlocal current_lines

        section_text = "\n".join(current_lines).strip()

        if not section_text:

            current_lines = []

            return

        blocks = split_markdown_blocks(section_text)

        chunks = pack_blocks(
            blocks,
            tokenizer,
            MAX_CHUNK_TOKENS,
            OVERLAP_TOKENS,
        )

        for chunk in chunks:

            sections.append(
                {
                    "heading_path": (current_heading_path or "Начало документа"),
                    "text": chunk,
                }
            )

        current_lines = []

    for line in text.splitlines():

        heading_match = HEADING_RE.match(line)

        if not heading_match:

            current_lines.append(line)

            continue

        flush_section()

        level = len(heading_match.group(1))

        heading = clean_heading(heading_match.group(2))

        heading_stack[:] = heading_stack[: level - 1]

        heading_stack.append(heading)

        current_heading_path = " → ".join(heading_stack)

    flush_section()

    return sections


def make_content_hash(
    text: str,
) -> str:
    """
    Хеш содержимого.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_point_id(
    relative_path: str,
    chunk_index: int,
    chunk_hash: str,
) -> str:
    """
    Создаёт стабильный UUID.
    """

    value = f"{relative_path}:" f"{chunk_index}:" f"{chunk_hash}"

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )
    )


# ============================================================
# ЧТЕНИЕ MARKDOWN
# ============================================================


def load_documents(
    tokenizer: TextTokenizer,
    max_sequence_tokens: int,
) -> list[dict]:

    markdown_files = sorted(MD_ROOT.rglob("*.md"))

    if not markdown_files:

        raise RuntimeError(f"В {MD_ROOT} не найдено MD-файлов")

    print(f"Найдено MD-файлов: " f"{len(markdown_files)}")

    documents: list[dict] = []

    for md_path in tqdm(
        markdown_files,
        desc="Разбор Markdown",
    ):

        try:

            raw_text = md_path.read_text(
                encoding="utf-8",
            )

        except UnicodeDecodeError:

            raw_text = md_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        text = remove_frontmatter(raw_text)

        relative_path = str(md_path.relative_to(MD_ROOT))

        title = first_h1(text) or md_path.stem

        sections = split_into_sections(
            text,
            tokenizer,
        )

        chunk_index = 0

        for section in sections:

            chunk_text = (section["text"]).strip()

            if len(chunk_text) < 30:
                continue

            heading_path = section["heading_path"]

            embedding_prefix = f"Документ: {title}\n" f"Раздел: {heading_path}\n\n"

            prefix_tokens = count_tokens(
                embedding_prefix,
                tokenizer,
                add_special_tokens=True,
            )

            available_tokens = min(
                MAX_CHUNK_TOKENS,
                max_sequence_tokens - prefix_tokens - TOKEN_SAFETY_MARGIN,
            )

            if available_tokens <= 0:
                raise RuntimeError(
                    "Название документа и путь раздела не помещаются "
                    f"в окно embedding-модели: {relative_path} / {heading_path}"
                )

            chunk_parts = split_long_text(
                chunk_text,
                tokenizer,
                available_tokens,
            )

            for chunk_part in chunk_parts:

                # Именно этот текст отправляется embedding-модели.
                # Заголовок документа и путь раздела учитываются
                # в общем токенном лимите.
                embedding_text = f"{embedding_prefix}{chunk_part}"

                embedding_tokens = count_tokens(
                    embedding_text,
                    tokenizer,
                    add_special_tokens=True,
                )

                if embedding_tokens > max_sequence_tokens:
                    raise RuntimeError(
                        "Чанк превышает окно embedding-модели: "
                        f"{embedding_tokens} > {max_sequence_tokens}, "
                        f"{relative_path} / {heading_path}"
                    )

                chunk_hash = make_content_hash(embedding_text)

                point_id = create_point_id(
                    relative_path,
                    chunk_index,
                    chunk_hash,
                )

                documents.append(
                    {
                        "id": point_id,
                        "embedding_text": embedding_text,
                        "payload": {
                            "title": title,
                            "heading_path": heading_path,
                            "text": chunk_part,
                            "source_path": relative_path,
                            "absolute_path": str(md_path),
                            "chunk_index": chunk_index,
                            "content_hash": chunk_hash,
                            "embedding_model": MODEL_NAME,
                        },
                    }
                )

                chunk_index += 1

    return documents


# ============================================================
# СОЗДАНИЕ ЭМБЕДДИНГОВ
# ============================================================


def main() -> None:

    print(f"Папка с Markdown:\n" f"{MD_ROOT}\n")

    if not MD_ROOT.exists():

        raise RuntimeError(f"Папка не существует: " f"{MD_ROOT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Устройство: {device}")

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(f"\nЗагрузка модели:\n" f"{MODEL_NAME}\n")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    vector_size = model.get_sentence_embedding_dimension()

    if vector_size is None:

        raise RuntimeError("Не удалось определить " "размер эмбеддинга")

    print(f"Размер эмбеддинга: " f"{vector_size}")

    max_sequence_tokens = model.max_seq_length

    if max_sequence_tokens is None:
        raise RuntimeError("Embedding-модель не сообщает max_seq_length")

    print(f"Максимум токенов модели: {max_sequence_tokens}")
    print(f"Целевой размер чанка: {MAX_CHUNK_TOKENS}")
    print(f"Перекрытие чанков: {OVERLAP_TOKENS}")

    documents = load_documents(
        model.tokenizer,
        max_sequence_tokens,
    )

    print(f"\nПодготовлено чанков: " f"{len(documents)}")

    if not documents:

        raise RuntimeError("Не удалось создать ни одного чанка")

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=120,
    )

    print(f"\nПодключение к Qdrant:\n" f"{QDRANT_URL}")

    # Для первой версии каждый запуск
    # полностью пересоздаёт индекс.
    #
    # Так проще:
    # не останутся старые чанки
    # после удаления или изменения MD.
    if client.collection_exists(COLLECTION_NAME):

        print(
            "Удаление старой коллекции:",
            COLLECTION_NAME,
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME,
        )

    print(
        "Создание коллекции:",
        COLLECTION_NAME,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    total = len(documents)

    for start in tqdm(
        range(
            0,
            total,
            QDRANT_BATCH_SIZE,
        ),
        desc="Создание эмбеддингов",
    ):

        batch = documents[start : start + QDRANT_BATCH_SIZE]

        texts = [item["embedding_text"] for item in batch]

        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        points: list[PointStruct] = []

        for item, vector in zip(
            batch,
            embeddings,
        ):

            points.append(
                PointStruct(
                    id=item["id"],
                    vector=vector.tolist(),
                    payload=item["payload"],
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    collection = client.get_collection(COLLECTION_NAME)

    print("\n================================")

    print("Индексация завершена")

    print(
        "Коллекция:",
        COLLECTION_NAME,
    )

    print(
        "Чанков:",
        len(documents),
    )

    print(
        "Размер вектора:",
        vector_size,
    )

    print(
        "Статус:",
        collection.status,
    )

    print("================================")


if __name__ == "__main__":
    main()
