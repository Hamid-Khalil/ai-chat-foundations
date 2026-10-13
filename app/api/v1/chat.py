"""All /chat/* endpoints. Thin: validate -> call ONE service function -> return."""

from typing import Any

from fastapi import APIRouter

from app.api.deps import SettingsDep
from app.schemas.chat import (
    AskRequest,
    ChatReply,
    ConversationRequest,
    ErrorResponse,
    ModelInfo,
)
from app.services import chat_service

# No prefix here: "/chat" is assigned once, in api/v1/router.py.
router = APIRouter()

_LLM_ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"description": "Validation error (FastAPI default shape)"},
    429: {
        "model": ErrorResponse,
        "description": "LLM_RATE_LIMITED - Gemini free-tier quota exceeded",
    },
    502: {
        "model": ErrorResponse,
        "description": "LLM_UPSTREAM_ERROR or LLM_EMPTY_RESPONSE - Gemini failed",
    },
    504: {
        "model": ErrorResponse,
        "description": "LLM_TIMEOUT - Gemini did not answer in time",
    },
}


@router.post(
    "/ask",
    tags=["Simple Chat"],
    response_model=ChatReply,
    summary="Ask the study buddy one question",
    description=(
        "Sends one question to Gemini together with the server-side system prompt "
        "and returns the answer, token usage and finish reason. Nothing is "
        "remembered between calls."
    ),
    responses=_LLM_ERRORS,
)
async def ask(body: AskRequest) -> ChatReply:
    return await chat_service.ask(body.message, body.thinking_level)


@router.post(
    "/messages",
    tags=["Conversation Chat"],
    response_model=ChatReply,
    summary="Continue a conversation",
    description=(
        "The client re-sends the whole history (user/assistant turns, alternating, "
        "starting and ending with 'user'). The server adds the only system message "
        "and returns the next assistant reply."
    ),
    responses=_LLM_ERRORS,
)
async def messages(body: ConversationRequest) -> ChatReply:
    return await chat_service.converse(body.messages, body.thinking_level)


@router.get(
    "/model",
    tags=["Model Settings"],
    response_model=ModelInfo,
    summary="Show the model settings the server is running with",
    description=(
        "Returns provider, model, default thinking level and limits from the "
        "server's configuration. The API key is never included. Does not call Gemini."
    ),
    responses={500: {"model": ErrorResponse, "description": "INTERNAL_ERROR"}},
)
async def model_info(settings: SettingsDep) -> ModelInfo:
    return chat_service.get_model_info(settings)
