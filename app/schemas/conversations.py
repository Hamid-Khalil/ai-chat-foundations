from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class UsageTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ConversationOut(BaseModel):
    id: str
    title: str
    archived: bool
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: str
    usage_totals: UsageTotals

    @field_serializer("created_at", "updated_at", check_fields=False)
    def serialize_dt(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ConversationList(BaseModel):
    items: List[ConversationOut]
    next_cursor: Optional[str] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=100)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def strip_title(self) -> "ConversationCreate":
        if self.title is not None:
            self.title = self.title.strip()
            if not self.title:
                raise ValueError("title cannot be blank")
        return self


class ConversationUpdate(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{"rating": "up", "comment": "Great answer, very helpful!"}]
        },
    }
    title: Optional[str] = Field(default=None, min_length=1, max_length=100)
    archived: Optional[bool] = None

    @model_validator(mode="after")
    def check_at_least_one(self) -> "ConversationUpdate":
        if self.title is None and self.archived is None:
            raise ValueError("At least one of 'title' or 'archived' must be provided")
        if self.title is not None:
            self.title = self.title.strip()
            if not self.title:
                raise ValueError("title cannot be blank")
        return self


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    seq: int
    role: Literal["user", "assistant"]
    content: str
    mode: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    feedback: Optional[Dict[str, Any]] = None

    @field_serializer("created_at", check_fields=False)
    def serialize_dt(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class MessageList(BaseModel):
    items: List[MessageOut]
    next_after_seq: Optional[int] = None


class DeleteResponse(BaseModel):
    conversation_id: str
    deleted_messages: int


from app.core.config import get_settings


class PersistedChatRequest(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{"rating": "up", "comment": "Great answer, very helpful!"}]
        },
    }
    content: str = Field(..., min_length=1)
    mode: Literal["chat", "tools", "rag"] = "chat"
    thinking_level: Optional[str] = None
    tools: Optional[List[str]] = None
    top_k: Optional[int] = None
    document_ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "PersistedChatRequest":
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("content cannot be blank")

        settings = get_settings()

        if self.mode == "chat":
            if (
                self.tools is not None
                or self.top_k is not None
                or self.document_ids is not None
            ):
                raise ValueError(
                    "tools, top_k, and document_ids are not allowed in 'chat' mode"
                )
            if len(self.content) > settings.chat_max_message_chars:
                raise ValueError(
                    f"content exceeds {settings.chat_max_message_chars} characters"
                )

        elif self.mode == "tools":
            if self.top_k is not None or self.document_ids is not None:
                raise ValueError(
                    "top_k and document_ids are not allowed in 'tools' mode"
                )
            if not self.tools or len(self.tools) < 1:
                raise ValueError("At least one tool must be provided in 'tools' mode")
            if len(self.content) > settings.chat_max_message_chars:
                raise ValueError(
                    f"content exceeds {settings.chat_max_message_chars} characters"
                )

        elif self.mode == "rag":
            if self.tools is not None:
                raise ValueError("tools is not allowed in 'rag' mode")
            rag_max = getattr(settings, "rag_top_k_max", 10)
            if self.top_k is not None and (self.top_k < 1 or self.top_k > rag_max):
                raise ValueError(f"top_k must be between 1 and {rag_max}")
            if self.document_ids is not None:
                if len(self.document_ids) < 1 or len(self.document_ids) > 20:
                    raise ValueError("document_ids must contain 1 to 20 items")
                if any(len(doc_id) > 64 for doc_id in self.document_ids):
                    raise ValueError("document_ids items must be at most 64 characters")
            rag_chars = getattr(settings, "rag_max_question_chars", 1000)
            if len(self.content) > rag_chars:
                raise ValueError(
                    f"content exceeds maximum length ({rag_chars} characters) for rag mode"
                )

        return self


class PersistedChatResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


class FeedbackIn(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{"rating": "up", "comment": "Great answer, very helpful!"}]
        },
    }
    rating: Literal["up", "down"]
    comment: Optional[str] = None

    @field_validator("comment", mode="before")
    @classmethod
    def validate_comment(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Comment must be a string")
        stripped = v.strip()
        if not stripped:
            return None
        if len(stripped) > 500:
            raise ValueError("Comment must be at most 500 characters")
        return stripped


class FeedbackDetail(BaseModel):
    rating: Literal["up", "down"]
    comment: Optional[str] = None
    created_at: datetime


class ConversationStats(BaseModel):
    active: int = 0
    archived: int = 0


class MessageStats(BaseModel):
    total: int = 0
    by_mode: dict[str, int] = Field(default_factory=dict)
    by_role: dict[str, int] = Field(default_factory=dict)


class TokenStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class FeedbackStats(BaseModel):
    up: int = 0
    down: int = 0


class StatsOut(BaseModel):
    conversations: ConversationStats
    messages: MessageStats
    tokens: TokenStats
    feedback: FeedbackStats
