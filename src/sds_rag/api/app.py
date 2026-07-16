import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from ..core.config import (
    API_VERSION,
    COLLECTION_NAME,
    CONTEXT_SAFETY_MARGIN_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DENSE_VECTOR_NAME,
    EMBEDDING_MODEL,
    HYBRID_PREFETCH_LIMIT,
    LLM_MODEL,
    LLM_URL,
    MAX_CONTEXT_ITEM_TOKENS,
    MAX_CONTEXTS,
    MAX_HISTORY_MESSAGES,
    MAX_HISTORY_MESSAGE_TOKENS,
    MAX_HISTORY_TOKENS,
    MAX_OUTPUT_TOKENS,
    MIN_CONTEXT_ITEM_TOKENS,
    MIN_OUTPUT_TOKENS,
    MODEL_CONTEXT_TOKENS,
    QDRANT_URL,
    RAG_API_KEY,
    RAG_MODEL_ID,
    RAG_MODEL_NAME,
    RETRIEVAL_LIMIT,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_LANGUAGE,
    SPARSE_VECTOR_NAME,
    TOKENIZER_MODEL,
    WIKI_BASE_URL,
)
from ..core.helpers import (
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
from .schemas import (
    AskRequest,
    AskResponse,
    ChatCompletionRequest,
    ChatMessage,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Source,
)

# ============================================================
# ЗАГРУЗКА МОДЕЛИ И КЛИЕНТОВ
# ============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Устройство dense-модели: " f"{device}")

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(f"Загрузка dense-модели: " f"{EMBEDDING_MODEL}")

    app.state.embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )

    print(f"Загрузка токенизатора LLM: " f"{TOKENIZER_MODEL}")

    app.state.llm_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"Загрузка sparse-модели: " f"{SPARSE_EMBEDDING_MODEL}")

    try:
        app.state.sparse_model = SparseTextEmbedding(
            model_name=SPARSE_EMBEDDING_MODEL,
            language=SPARSE_LANGUAGE,
        )

        print(
            "Язык BM25:",
            SPARSE_LANGUAGE,
        )

    except TypeError as exc:
        raise RuntimeError(
            "Установленная версия FastEmbed "
            "не поддерживает параметр language. "
            "Обновите зависимость командой: "
            'uv add "qdrant-client[fastembed]>=1.14.2"'
        ) from exc

    app.state.qdrant = QdrantClient(
        url=QDRANT_URL,
        timeout=60,
    )

    if not app.state.qdrant.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            "Hybrid-коллекция Qdrant " f"'{COLLECTION_NAME}' " "не найдена"
        )

    print("Wiki.js Hybrid RAG API запущен")

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Wiki.js Hybrid RAG API",
    description=(
        "Hybrid RAG по документации Wiki.js: "
        "ai-forever/FRIDA dense + BM25 sparse + RRF "
        "+ OpenAI-compatible API "
        "для Open WebUI"
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


def make_wiki_url(
    source_path: str,
) -> str:
    path = source_path.replace(
        "\\",
        "/",
    )

    if path.endswith(".md"):
        path = path[:-3]

    encoded_path = quote(
        path,
        safe="/",
    )

    return f"{WIKI_BASE_URL.rstrip('/')}" f"/{encoded_path.lstrip('/')}"


def encode_text(
    text: str,
) -> list[int]:
    tokenizer = app.state.llm_tokenizer

    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def count_text_tokens(
    text: str,
) -> int:
    if not text:
        return 0

    return len(encode_text(text))


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    keep_tail: bool = False,
) -> str:
    """
    Обрезает текст по реальным токенам Qwen,
    а не по символам.

    Для больших разделов можно сохранить
    начало и конец, чтобы не потерять
    финальные примечания и условия.
    """

    if not text or max_tokens <= 0:
        return ""

    tokenizer = app.state.llm_tokenizer

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(token_ids) <= max_tokens:
        return text

    if not keep_tail or max_tokens < 96:
        return tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=True,
        ).strip()

    marker = "\n\n" "[... часть большого раздела сокращена ...]" "\n\n"

    marker_ids = tokenizer.encode(
        marker,
        add_special_tokens=False,
    )

    available = max_tokens - len(marker_ids)

    if available < 64:
        return tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=True,
        ).strip()

    head_size = int(available * 0.75)

    tail_size = available - head_size

    return (
        tokenizer.decode(
            token_ids[:head_size],
            skip_special_tokens=True,
        ).strip()
        + marker
        + tokenizer.decode(
            token_ids[-tail_size:],
            skip_special_tokens=True,
        ).strip()
    )


def count_chat_tokens(
    messages: list[dict[str, str]],
) -> int:
    """
    Считает prompt уже с chat template Qwen.

    Это важнее подсчёта символов: vLLM ограничивает
    сумму входных и выходных токенов.
    """

    tokenizer = app.state.llm_tokenizer

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if isinstance(
            tokenized,
            torch.Tensor,
        ):
            return int(tokenized.numel())

        if tokenized and isinstance(
            tokenized[0],
            list,
        ):
            return len(tokenized[0])

        return len(tokenized)

    except Exception:
        # Консервативный fallback для нестандартного
        # токенизатора или chat template.
        return (
            sum(
                count_text_tokens(
                    message.get(
                        "content",
                        "",
                    )
                )
                + 16
                for message in messages
            )
            + 32
        )


def verify_openai_key(
    authorization: str | None,
) -> None:
    expected = f"Bearer {RAG_API_KEY}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )


def select_returned_sources(
    answer: str,
    sources: list[Source],
) -> list[Source]:
    """
    Возвращает только реально процитированные источники.

    Важно:
    раньше при отсутствии цитат возвращался sources[:1],
    поэтому Open WebUI показывал случайный первый документ
    даже для ответа «информация не найдена».
    """

    if is_no_answer(answer):
        return []

    citation_numbers = extract_cited_source_numbers(answer)

    if not citation_numbers:
        return []

    by_number = {source.citation_number: source for source in sources}

    return [by_number[number] for number in citation_numbers if number in by_number]


def make_history_text(
    messages: list[ChatMessage],
) -> str:
    """
    Собирает историю с конца и укладывает её
    в отдельный токен-бюджет.

    Полная история Open WebUI больше никогда
    не отправляется в LLM целиком.
    """

    selected_reversed: list[str] = []
    used_tokens = 0

    recent_messages = messages[-MAX_HISTORY_MESSAGES:]

    for message in reversed(recent_messages):
        role = message.role.strip().lower()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        text = clean_history_message(content_to_text(message.content))

        if not text:
            continue

        text = truncate_text_to_tokens(
            text,
            MAX_HISTORY_MESSAGE_TOKENS,
            keep_tail=False,
        )

        label = "Пользователь" if role == "user" else "Ассистент"

        entry = f"{label}: {text}"

        entry_tokens = count_text_tokens(entry)

        remaining = MAX_HISTORY_TOKENS - used_tokens

        if remaining <= 0:
            break

        if entry_tokens > remaining:
            if remaining < 48:
                break

            entry = f"{label}: " + truncate_text_to_tokens(
                text,
                max(
                    16,
                    remaining - 8,
                ),
                keep_tail=False,
            )

            entry_tokens = count_text_tokens(entry)

        selected_reversed.append(entry)

        used_tokens += entry_tokens

    return "\n\n".join(reversed(selected_reversed))


def get_last_user_question(
    messages: list[ChatMessage],
) -> str:
    for message in reversed(messages):
        if message.role.strip().lower() != "user":
            continue

        text = content_to_text(message.content)

        if text:
            return text

    raise HTTPException(
        status_code=400,
        detail=("В messages нет " "сообщения пользователя"),
    )


def make_retrieval_query(
    messages: list[ChatMessage],
    current_question: str,
) -> str:
    """
    Для зависимого вопроса добавляет только
    предыдущий вопрос пользователя.

    Полный прошлый ответ в retrieval не попадает.
    """

    if not is_follow_up_question(current_question):
        return current_question

    previous_questions: list[str] = []

    for message in messages:
        if message.role.strip().lower() != "user":
            continue

        text = content_to_text(message.content)

        if text:
            previous_questions.append(text)

    if len(previous_questions) < 2:
        return current_question

    previous_question = previous_questions[-2]

    previous_question = truncate_text_to_tokens(
        previous_question,
        256,
        keep_tail=False,
    )

    return f"{previous_question}\n\n" f"Уточняющий вопрос:\n" f"{current_question}"


def make_sparse_query_vector(
    question: str,
) -> models.SparseVector:
    sparse_model = app.state.sparse_model

    embedding = next(iter(sparse_model.embed([question])))

    return sparse_embedding_to_vector(embedding)


def hybrid_query_points(
    question: str,
    limit: int,
):
    """
    Два независимых поиска:

    1. ai-forever/FRIDA dense — ищет по смыслу.
    2. BM25 sparse — ищет точные слова,
       коды, процедуры и идентификаторы.

    Qdrant объединяет ранги через RRF.
    """

    dense_model = app.state.embedding_model

    client = app.state.qdrant

    dense_vector = dense_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    sparse_vector = make_sparse_query_vector(question)

    prefetch_limit = max(
        HYBRID_PREFETCH_LIMIT,
        limit * 4,
    )

    return client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector.tolist(),
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )


def retrieve_context(
    question: str,
) -> tuple[
    list[str],
    list[Source],
]:
    response = hybrid_query_points(
        question=question,
        limit=RETRIEVAL_LIMIT,
    )

    points = response.points

    if not points:
        return [], []

    technical_identifiers = extract_technical_identifiers(question)

    if technical_identifiers:
        matching_points = []

        for point in points:
            payload = point.payload or {}

            if all(
                payload_contains_identifier(
                    payload,
                    identifier,
                )
                for identifier in technical_identifiers
            ):
                matching_points.append(point)

        # Для точного технического кода не используем
        # случайные семантически похожие документы.
        if not matching_points:
            return [], []

        points = matching_points

    contexts: list[str] = []
    sources: list[Source] = []

    used_sections: set[tuple[str, str]] = set()

    for point in points:
        current_score = float(point.score)

        payload = point.payload or {}

        title = str(
            payload.get(
                "title",
                "",
            )
        )

        heading_path = str(
            payload.get(
                "heading_path",
                "",
            )
        )

        text = str(
            payload.get(
                "text",
                "",
            )
        )

        source_path = str(
            payload.get(
                "source_path",
                "",
            )
        )

        section_key = (
            source_path,
            heading_path,
        )

        if section_key in used_sections:
            continue

        used_sections.add(section_key)

        source_number = len(contexts) + 1

        contexts.append(f"""
[Источник {source_number}]

Документ:
{title}

Раздел:
{heading_path}

Текст:
{text}
""".strip())

        sources.append(
            Source(
                citation_number=source_number,
                title=title,
                heading_path=heading_path,
                wiki_url=make_wiki_url(source_path),
                score=round(
                    current_score,
                    5,
                ),
            )
        )

        if len(contexts) >= RETRIEVAL_LIMIT:
            break

    return (
        contexts,
        sources,
    )


SYSTEM_PROMPT = """
Ты помощник по внутренней технической документации SDS.

Отвечай только на основании переданных источников.

История диалога нужна только для понимания
контекста и ссылок вроде «это», «там»,
«а как дальше». Не используй историю
как источник технических фактов.

Не придумывай настройки, параметры,
адреса, названия процедур,
значения полей и причины ошибок.

Если информации недостаточно, прямо напиши:
«В документации недостаточно информации
для однозначного ответа».

Если информации недостаточно,
не указывай [Источник N]
и не придумывай ссылку на источник.

Отвечай по-русски.

Для простого вопроса дай краткий ответ
в 2–4 абзацах.

Для инструкции используй
понятные нумерованные шаги.

Не пересказывай весь найденный документ.
Выбирай только сведения,
необходимые для ответа.

После каждого логического абзаца
или группы шагов указывай
реально использованный источник
строго в формате:

[Источник 1]

Не копируй сырые внутренние Markdown-ссылки
вида /infoclinica/... и HTML-теги.
Ссылки будут добавлены интерфейсом отдельно.
""".strip()


def build_user_prompt(
    question: str,
    history_text: str,
    contexts: list[str],
) -> str:
    history_block = history_text if history_text else "История отсутствует."

    context_text = (
        "\n\n".join(contexts) if contexts else "Подходящие источники не найдены."
    )

    return f"""
История диалога:

{history_block}


Текущий вопрос:

{question}


Документация:

{context_text}
""".strip()


def build_chat_messages(
    question: str,
    history_text: str,
    contexts: list[str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                question,
                history_text,
                contexts,
            ),
        },
    ]


def normalize_output_tokens(
    requested: int | None,
) -> int:
    value = DEFAULT_MAX_OUTPUT_TOKENS if requested is None else requested

    return min(
        max(
            int(value),
            MIN_OUTPUT_TOKENS,
        ),
        MAX_OUTPUT_TOKENS,
    )


def fit_context_block(
    block: str,
    token_limit: int,
) -> str:
    return truncate_text_to_tokens(
        block,
        token_limit,
        keep_tail=True,
    )


def prepare_generation_prompt(
    *,
    question: str,
    history_text: str,
    contexts: list[str],
    sources: list[Source],
    desired_output_tokens: int,
    max_contexts: int = MAX_CONTEXTS,
    max_context_item_tokens: int = (MAX_CONTEXT_ITEM_TOKENS),
) -> tuple[
    list[dict[str, str]],
    list[Source],
    int,
    int,
]:
    """
    Формирует prompt по реальному токен-бюджету.

    Инвариант:

    prompt_tokens
    + output_tokens
    + safety_margin
    <= MODEL_CONTEXT_TOKENS
    """

    max_input_tokens = (
        MODEL_CONTEXT_TOKENS - desired_output_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
    )

    # История уже ограничена отдельно, но если даже
    # базовый prompt не помещается — убираем её первой.
    base_messages = build_chat_messages(
        question,
        history_text,
        [],
    )

    base_tokens = count_chat_tokens(base_messages)

    if base_tokens > max_input_tokens:
        history_text = ""

        base_messages = build_chat_messages(
            question,
            history_text,
            [],
        )

        base_tokens = count_chat_tokens(base_messages)

    selected_contexts: list[str] = []
    selected_sources: list[Source] = []

    for context, source in zip(
        contexts,
        sources,
    ):
        if len(selected_contexts) >= max_contexts:
            break

        current_messages = build_chat_messages(
            question,
            history_text,
            selected_contexts,
        )

        current_tokens = count_chat_tokens(current_messages)

        remaining = max_input_tokens - current_tokens

        if remaining < MIN_CONTEXT_ITEM_TOKENS:
            break

        upper_limit = min(
            max_context_item_tokens,
            max(
                MIN_CONTEXT_ITEM_TOKENS,
                remaining - 24,
            ),
        )

        candidate = fit_context_block(
            context,
            upper_limit,
        )

        trial_contexts = selected_contexts + [candidate]

        trial_messages = build_chat_messages(
            question,
            history_text,
            trial_contexts,
        )

        trial_tokens = count_chat_tokens(trial_messages)

        if trial_tokens <= max_input_tokens:
            selected_contexts.append(candidate)

            selected_sources.append(source)

            continue

        # Точный бинарный поиск по токенам
        # для последнего помещающегося куска.
        low = MIN_CONTEXT_ITEM_TOKENS

        high = upper_limit

        best_candidate: str | None = None

        while low <= high:
            middle = (low + high) // 2

            shortened = fit_context_block(
                context,
                middle,
            )

            shortened_messages = build_chat_messages(
                question,
                history_text,
                (selected_contexts + [shortened]),
            )

            shortened_tokens = count_chat_tokens(shortened_messages)

            if shortened_tokens <= max_input_tokens:
                best_candidate = shortened

                low = middle + 1

            else:
                high = middle - 1

        if best_candidate is None:
            break

        selected_contexts.append(best_candidate)

        selected_sources.append(source)

    messages = build_chat_messages(
        question,
        history_text,
        selected_contexts,
    )

    prompt_tokens = count_chat_tokens(messages)

    available_output_tokens = (
        MODEL_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
    )

    actual_output_tokens = min(
        desired_output_tokens,
        available_output_tokens,
    )

    if actual_output_tokens < MIN_OUTPUT_TOKENS:
        # Это аварийная страховка. Обычно сюда
        # не попадём, потому что контексты пакуются
        # уже с учётом desired_output_tokens.
        while selected_contexts and actual_output_tokens < MIN_OUTPUT_TOKENS:
            selected_contexts.pop()
            selected_sources.pop()

            messages = build_chat_messages(
                question,
                history_text,
                selected_contexts,
            )

            prompt_tokens = count_chat_tokens(messages)

            available_output_tokens = (
                MODEL_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
            )

            actual_output_tokens = min(
                desired_output_tokens,
                available_output_tokens,
            )

    if actual_output_tokens < 64:
        raise HTTPException(
            status_code=500,
            detail=("Не удалось сформировать prompt " "в пределах контекстного окна"),
        )

    return (
        messages,
        selected_sources,
        int(actual_output_tokens),
        int(prompt_tokens),
    )


async def call_vllm(
    *,
    messages: list[dict[str, str]],
    output_tokens: int,
    temperature: float | None,
) -> httpx.Response:
    request_json = {
        "model": LLM_MODEL,
        "temperature": (
            0.1
            if temperature is None
            else min(
                max(
                    temperature,
                    0.0,
                ),
                1.0,
            )
        ),
        "max_tokens": output_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "messages": messages,
    }

    async with httpx.AsyncClient(
        timeout=240,
    ) as http:
        return await http.post(
            f"{LLM_URL}" f"/chat/completions",
            json=request_json,
        )


async def generate_rag_answer(
    question: str,
    contexts: list[str],
    sources: list[Source],
    history_text: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[
    str,
    list[Source],
]:
    if not contexts:
        return (
            "В документации не найдено "
            "достаточно релевантных сведений "
            "для ответа.",
            [],
        )

    desired_output_tokens = normalize_output_tokens(max_tokens)

    (
        messages,
        used_sources,
        actual_output_tokens,
        prompt_tokens,
    ) = prepare_generation_prompt(
        question=question,
        history_text=history_text,
        contexts=contexts,
        sources=sources,
        desired_output_tokens=desired_output_tokens,
    )

    print(
        "RAG token budget | "
        f"prompt={prompt_tokens} | "
        f"output={actual_output_tokens} | "
        f"reserve={CONTEXT_SAFETY_MARGIN_TOKENS} | "
        f"contexts={len(used_sources)}"
    )

    try:
        response = await call_vllm(
            messages=messages,
            output_tokens=actual_output_tokens,
            temperature=temperature,
        )

        # Последняя страховка от несовпадения
        # локального и серверного chat template.
        if response.status_code == 400 and (
            "maximum context length" in response.text.lower()
            or "context length" in response.text.lower()
        ):
            (
                messages,
                used_sources,
                actual_output_tokens,
                prompt_tokens,
            ) = prepare_generation_prompt(
                question=question,
                history_text="",
                contexts=contexts,
                sources=sources,
                desired_output_tokens=min(
                    desired_output_tokens,
                    512,
                ),
                max_contexts=3,
                max_context_item_tokens=900,
            )

            print(
                "RAG emergency retry | "
                f"prompt={prompt_tokens} | "
                f"output={actual_output_tokens} | "
                f"contexts={len(used_sources)}"
            )

            response = await call_vllm(
                messages=messages,
                output_tokens=actual_output_tokens,
                temperature=temperature,
            )

        response.raise_for_status()

    except (
        httpx.ConnectError,
        httpx.TimeoutException,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=("Не удалось получить " "ответ от vLLM"),
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "vLLM вернул ошибку: "
                f"{exc.response.status_code} "
                f"{exc.response.text[:500]}"
            ),
        ) from exc

    data = response.json()

    try:
        answer = str(data["choices"][0]["message"]["content"]).strip()

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=("Неожиданный формат " "ответа от vLLM"),
        ) from exc

    return (
        answer,
        used_sources,
    )


def rewrite_relative_wiki_links(
    text: str,
) -> str:
    """
    Open WebUI открывается на :3001.

    Поэтому относительная Markdown-ссылка:

    [прейскурант](/infoclinica/...)

    без обработки открылась бы как:

    http://localhost:3001/infoclinica/...

    Здесь все относительные ссылки Wiki.js
    превращаются в абсолютные ссылки на :3000
    или на адрес из WIKI_BASE_URL.
    """

    def replace_markdown_link(
        match: re.Match,
    ) -> str:
        label = match.group(1)
        url = match.group(2).strip()

        # Якоря и уже абсолютные ссылки
        # оставляем без изменений.
        if (
            url.startswith("#")
            or url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("mailto:")
        ):
            return match.group(0)

        if url.startswith("/"):
            absolute_url = f"{WIKI_BASE_URL.rstrip('/')}" f"{url}"
        else:
            absolute_url = f"{WIKI_BASE_URL.rstrip('/')}/" f"{url.lstrip('/')}"

        return f"[{label}]" f"({absolute_url})"

    return re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        replace_markdown_link,
        text,
    )


def format_for_openwebui(
    answer: str,
    sources: list[Source],
) -> str:
    """
    1. Исправляет относительные ссылки Wiki.js.
    2. Превращает [Источник N]
       в кликабельные ссылки.
    3. Добавляет блок источников.
    """

    answer = rewrite_relative_wiki_links(answer)

    by_number = {source.citation_number: source for source in sources}

    def replace_citation(
        match: re.Match,
    ) -> str:
        number = int(match.group(1))

        source = by_number.get(number)

        if source is None:
            return f"[Источник {number}]"

        return f"[Источник {number}]" f"({source.wiki_url})"

    linked_answer = re.sub(
        r"\[Источник\s+(\d+)\]",
        replace_citation,
        answer,
        flags=re.IGNORECASE,
    )

    selected_sources = select_returned_sources(
        answer,
        sources,
    )

    if not selected_sources:
        return linked_answer

    source_lines = []

    for source in selected_sources:
        label = source.title

        if source.heading_path and source.heading_path != source.title:
            label += " — " + source.heading_path

        source_lines.append(f"- [{label}]" f"({source.wiki_url})")

    return f"{linked_answer}\n\n" f"---\n" f"### Источники\n" + "\n".join(source_lines)


# ============================================================
# ОБЫЧНЫЕ ENDPOINTS
# ============================================================


@app.get("/")
def root() -> dict:
    return {
        "service": "Wiki.js Hybrid RAG API",
        "version": API_VERSION,
        "docs": "/docs",
        "openai_models": "/v1/models",
    }


@app.get("/health")
def health() -> dict:
    collection = app.state.qdrant.get_collection(COLLECTION_NAME)

    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "points_count": collection.points_count,
        "retrieval": "hybrid_rrf",
        "dense_embedding_model": EMBEDDING_MODEL,
        "sparse_embedding_model": SPARSE_EMBEDDING_MODEL,
        "dense_vector_name": DENSE_VECTOR_NAME,
        "sparse_vector_name": SPARSE_VECTOR_NAME,
        "llm_model": LLM_MODEL,
        "openwebui_model": RAG_MODEL_ID,
        "context_management": "token_budget",
        "model_context_tokens": MODEL_CONTEXT_TOKENS,
        "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
        "max_history_tokens": MAX_HISTORY_TOKENS,
        "max_context_item_tokens": MAX_CONTEXT_ITEM_TOKENS,
    }


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
) -> SearchResponse:
    response = hybrid_query_points(
        question=request.question,
        limit=request.limit,
    )

    results: list[SearchResult] = []

    for point in response.points:
        payload = point.payload or {}

        source_path = str(
            payload.get(
                "source_path",
                "",
            )
        )

        results.append(
            SearchResult(
                score=round(
                    float(point.score),
                    5,
                ),
                title=str(
                    payload.get(
                        "title",
                        "",
                    )
                ),
                heading_path=str(
                    payload.get(
                        "heading_path",
                        "",
                    )
                ),
                text=str(
                    payload.get(
                        "text",
                        "",
                    )
                ),
                source_path=source_path,
                wiki_url=make_wiki_url(source_path),
            )
        )

    return SearchResponse(
        question=request.question,
        count=len(results),
        results=results,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(
    request: AskRequest,
) -> AskResponse:
    contexts, sources = retrieve_context(request.question)

    (
        answer,
        used_sources,
    ) = await generate_rag_answer(
        question=request.question,
        contexts=contexts,
        sources=sources,
    )

    return AskResponse(
        answer=answer,
        sources=select_returned_sources(
            answer,
            used_sources,
        ),
    )


# ============================================================
# OPENAI-COMPATIBLE ENDPOINTS ДЛЯ OPEN WEBUI
# ============================================================


@app.get("/v1/models")
def openai_models(
    authorization: str | None = Header(default=None),
) -> dict:
    verify_openai_key(authorization)

    return {
        "object": "list",
        "data": [
            {
                "id": RAG_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sds",
                "name": RAG_MODEL_NAME,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    verify_openai_key(authorization)

    question = get_last_user_question(request.messages)

    retrieval_query = make_retrieval_query(
        request.messages,
        question,
    )

    contexts, sources = retrieve_context(retrieval_query)

    history_text = (
        make_history_text(request.messages[:-1])
        if is_follow_up_question(question)
        else ""
    )

    (
        answer,
        used_sources,
    ) = await generate_rag_answer(
        question=question,
        contexts=contexts,
        sources=sources,
        history_text=history_text,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    content = format_for_openwebui(
        answer,
        used_sources,
    )

    completion_id = "chatcmpl-" + uuid.uuid4().hex

    created = int(time.time())

    response_model = request.model or RAG_MODEL_ID

    if not request.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def event_stream():
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None,
                }
            ],
        }

        yield (
            "data: "
            + json.dumps(
                first_chunk,
                ensure_ascii=False,
            )
            + "\n\n"
        )

        for piece in split_stream_text(content):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": piece,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            await asyncio.sleep(0)

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        yield (
            "data: "
            + json.dumps(
                final_chunk,
                ensure_ascii=False,
            )
            + "\n\n"
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
