"""Schemas for RAG generation and endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings
from app.schemas.chat import ChatReply, ConversationRequest, Usage


class RagAskRequest(BaseModel):
    """Payload for POST /rag/ask."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the lodging expense ceiling per night in standard cities?",
                "top_k": 4,
                "document_ids": None,
                "thinking_level": "low",
            }
        }
    )

    question: str = Field(..., min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=8)
    document_ids: list[str] | None = Field(default=None, max_length=20)
    thinking_level: Literal["low", "medium", "high"] = "low"


class SourceChunk(BaseModel):
    """A single chunk of text provided to the model as context."""

    source_id: int
    document_id: str
    filename: str
    chunk_index: int
    score: float
    snippet: str
    cited: bool = False
    page: int | None = None


class RetrievalInfo(BaseModel):
    """Metadata about the retrieval step."""

    top_k: int
    min_score: float
    retrieved: int
    used: int


class RagReply(ChatReply):
    """RAG-augmented chat response."""

    answered_from_context: bool
    answer_found: bool
    sources: list[SourceChunk]
    retrieval: RetrievalInfo
    standalone_question: str | None = None


class RagMessagesRequest(ConversationRequest):
    """Payload for POST /rag/messages."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "How many days of paid leave do full-time employees receive?",
                    },
                    {
                        "role": "assistant",
                        "content": "Full-time employees receive 27 days of annual leave [1].",
                    },
                    {"role": "user", "content": "What about contractors?"},
                ],
                "top_k": 4,
                "document_ids": None,
            }
        }
    )

    top_k: int | None = Field(default=None, ge=1, le=8)
    document_ids: list[str] | None = Field(default=None, max_length=20)

    @field_validator("messages")
    @classmethod
    def _validate_last_message_length(cls, messages):
        max_chars = get_settings().rag_max_question_chars
        if len(messages[-1].content) > max_chars:
            raise ValueError(
                f"Last user message must be at most {max_chars} characters."
            )
        return messages
