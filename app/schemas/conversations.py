from pydantic import BaseModel, Field, field_serializer, model_validator
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

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

    @model_validator(mode="after")
    def strip_title(self) -> "ConversationCreate":
        if self.title is not None:
            self.title = self.title.strip()
            if not self.title:
                raise ValueError("title cannot be blank")
        return self

class ConversationUpdate(BaseModel):
    model_config = {"extra": "forbid"}
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
