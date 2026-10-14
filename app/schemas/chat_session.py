"""Schemas for JSON-based chat session management."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateChatResponse(BaseModel):
    chat_id: str
    title: str
    created_at: str


class ChatSessionSummary(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ChatTurnRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    thinking_level: Literal["low", "medium", "high"] = "low"
    top_k: int | None = Field(default=None, ge=1, le=8)
    document_ids: list[str] | None = None


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    sources: list[dict[str, Any]] | None = None


class ChatSessionDetail(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[StoredMessage]
