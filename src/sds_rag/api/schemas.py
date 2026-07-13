"""Pydantic request and response models exposed by the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.config import RAG_MODEL_ID


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    score: float
    title: str
    heading_path: str
    text: str
    source_path: str
    wiki_url: str


class SearchResponse(BaseModel):
    question: str
    count: int
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)


class Source(BaseModel):
    citation_number: int
    title: str
    heading_path: str
    wiki_url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    # Open WebUI may send metadata, chat_id, stream_options and other fields.
    model_config = ConfigDict(extra="allow")

    model: str = RAG_MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
