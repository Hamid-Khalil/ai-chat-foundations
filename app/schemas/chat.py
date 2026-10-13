"""Pydantic v2 schemas for the chat endpoints (no LangChain types here)."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import ThinkingLevel, get_settings

# NOTE: "minimal" is intentionally NOT allowed. It returns an upstream error on
# Gemini 3.8 Flash, so we reject it with a 422 before Gemini is ever called.
ALLOWED_THINKING_LEVELS: list[str] = list(get_args(ThinkingLevel))


def _check_content(value: str) -> str:
    """Shared rule: trimmed (by the model config) and 1..CHAT_MAX_MESSAGE_CHARS."""
    max_chars = get_settings().chat_max_message_chars
    if not value:
        raise ValueError("must not be empty or whitespace only")
    if len(value) > max_chars:
        raise ValueError(f"must be at most {max_chars} characters (got {len(value)})")
    return value


class AskRequest(BaseModel):
    """One question in, one answer out."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "message": "What are the latest updates on the ICC Cricket World Cup?",
                    "thinking_level": "low",
                }
            ]
        },
    )

    message: str
    thinking_level: ThinkingLevel | None = Field(
        default=None, description="Optional. Omit to use the server default."
    )

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _check_content(value)


class ChatMessage(BaseModel):
    """One turn of the conversation sent by the client."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # "system" is never accepted: the server adds the only system message.
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        return _check_content(value)


class ConversationRequest(BaseModel):
    """The full history, re-sent on every call (the model itself is stateless)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "My name is Sara."},
                        {"role": "assistant", "content": "Nice to meet you, Sara!"},
                        {"role": "user", "content": "What is my name?"},
                    ],
                    "thinking_level": "low",
                }
            ]
        }
    )

    messages: list[ChatMessage]
    thinking_level: ThinkingLevel | None = Field(
        default=None, description="Optional. Omit to use the server default."
    )

    @field_validator("messages")
    @classmethod
    def _validate_history(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        max_messages = get_settings().chat_max_messages
        if not 1 <= len(messages) <= max_messages:
            raise ValueError(
                f"must contain between 1 and {max_messages} messages "
                f"(got {len(messages)})"
            )

        # Strict alternation starting with "user": even index -> user, odd -> assistant.
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError(
                    f"messages[{index}] has role '{message.role}' but must be "
                    f"'{expected}' (roles must alternate, starting with 'user')"
                )

        # The history must END with a user turn: the model answers the last human
        # message, and Google's Gemini 3.8 migration notes say to remove "prefilled"
        # model turns. Combined with alternation, this means an odd message count.
        if messages[-1].role != "user":
            raise ValueError(
                f"messages[{len(messages) - 1}] is the last message and must have "
                "role 'user'"
            )
        return messages


class Usage(BaseModel):
    """Token counts reported by the model provider for one call."""

    input_tokens: int
    output_tokens: int
    # null when the model does not report thinking tokens.
    reasoning_tokens: int | None = None
    total_tokens: int


class ChatReply(BaseModel):
    """Same shape for /chat/ask and /chat/messages."""

    reply: str
    model: str
    thinking_level: ThinkingLevel
    finish_reason: str | None = Field(
        default=None,
        description="Why the model stopped, e.g. STOP or MAX_TOKENS (cut off).",
    )
    usage: Usage


class ModelLimits(BaseModel):
    max_messages: int
    max_message_chars: int


class ModelInfo(BaseModel):
    """What the server is really running with. Never contains the API key."""

    provider: str
    model: str
    thinking_level: ThinkingLevel
    allowed_thinking_levels: list[str]
    max_tokens: int
    timeout_seconds: int
    max_retries: int
    limits: ModelLimits


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: str


class ErrorBody(BaseModel):
    code: str
    message: str
    status: int


class ErrorResponse(BaseModel):
    """The standard error shape (422 keeps FastAPI's default shape)."""

    error: ErrorBody
