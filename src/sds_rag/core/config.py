"""Environment-based configuration for the RAG service."""

from __future__ import annotations

import os

from .. import __version__

API_VERSION = __version__


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a documented default."""
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable or fail on an ambiguous value."""
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Переменная {name} должна содержать "
        "одно из значений: true/false, yes/no, on/off, 1/0"
    )


QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "wikijs_docs_hybrid")
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse")
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
SPARSE_LANGUAGE = os.getenv("SPARSE_LANGUAGE", "russian")
HYBRID_PREFETCH_LIMIT = _env_int("HYBRID_PREFETCH_LIMIT", 40)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ai-forever/FRIDA")
RERANK_ENABLED = _env_bool("RERANK_ENABLED", True)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "Qwen/Qwen3-Reranker-0.6B",
)
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 40)
RERANK_MAX_LENGTH = _env_int("RERANK_MAX_LENGTH", 768)
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 4)
RERANK_FAIL_OPEN = _env_bool("RERANK_FAIL_OPEN", True)

WIKI_BASE_URL = os.getenv("WIKI_BASE_URL", "http://localhost:3000")
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8001/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B-AWQ")
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", LLM_MODEL)

RAG_MODEL_ID = os.getenv("RAG_MODEL_ID", "sds-wiki-rag")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "Справочная система SDS")
RAG_API_KEY = os.getenv("RAG_API_KEY", "sk-local-rag-change-me")

RETRIEVAL_LIMIT = _env_int("RETRIEVAL_LIMIT", 20)
MAX_CONTEXTS = _env_int("MAX_CONTEXTS", 10)
MAX_CHUNKS_PER_SECTION = _env_int("MAX_CHUNKS_PER_SECTION", 2)
MODEL_CONTEXT_TOKENS = _env_int("MODEL_CONTEXT_TOKENS", 32_768)
DEFAULT_MAX_OUTPUT_TOKENS = _env_int("DEFAULT_MAX_OUTPUT_TOKENS", 2_000)
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 4_096)
MIN_OUTPUT_TOKENS = _env_int("MIN_OUTPUT_TOKENS", 256)
CONTEXT_SAFETY_MARGIN_TOKENS = _env_int("CONTEXT_SAFETY_MARGIN_TOKENS", 512)
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 4)
MAX_HISTORY_TOKENS = _env_int("MAX_HISTORY_TOKENS", 1_000)
MAX_HISTORY_MESSAGE_TOKENS = _env_int("MAX_HISTORY_MESSAGE_TOKENS", 400)
MAX_CONTEXT_ITEM_TOKENS = _env_int("MAX_CONTEXT_ITEM_TOKENS", 2_200)
MIN_CONTEXT_ITEM_TOKENS = _env_int("MIN_CONTEXT_ITEM_TOKENS", 220)
