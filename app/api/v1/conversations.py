from typing import Optional

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_conversation_repo, get_message_repo
from app.repositories.protocols import ConversationRepo, MessageRepo
from app.schemas.chat import ErrorResponse
from app.schemas.conversations import (
    ConversationCreate,
    ConversationList,
    ConversationOut,
    ConversationUpdate,
    DeleteResponse,
    FeedbackIn,
    MessageList,
    MessageOut,
    PersistedChatRequest,
    PersistedChatResponse,
    StatsOut,
)
from app.services import conversation_service

router = APIRouter()

DB_ERROR_RESPONSES = {
    503: {
        "model": ErrorResponse,
        "description": "Database unavailable (DB_UNAVAILABLE)",
    },
    500: {
        "model": ErrorResponse,
        "description": "Unexpected database error (DB_ERROR)",
    },
}


@router.post(
    "",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Conversations"],
    summary="Create a new conversation",
    responses=DB_ERROR_RESPONSES,
)
async def create_conversation(
    request: Optional[ConversationCreate] = None,
    repo: ConversationRepo = Depends(get_conversation_repo),
):
    title = request.title if request else None
    return await conversation_service.create_conversation(repo, title)


@router.get(
    "",
    response_model=ConversationList,
    tags=["Conversations"],
    summary="List conversations with keyset cursor pagination",
    responses={
        **DB_ERROR_RESPONSES,
        400: {
            "model": ErrorResponse,
            "description": "Malformed cursor (INVALID_CURSOR)",
        },
    },
)
async def list_conversations(
    archived: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    repo: ConversationRepo = Depends(get_conversation_repo),
):
    return await conversation_service.list_conversations(repo, archived, limit, cursor)


@router.get(
    "/stats",
    response_model=StatsOut,
    tags=["Conversations"],
    summary="Get system-wide usage statistics",
    description="Aggregates conversation counts, message modes, token usage, and feedback ratings using MongoDB pipelines.",
    status_code=status.HTTP_200_OK,
    responses={503: {"description": "Database unavailable."}},
)
async def get_stats(
    conversations: ConversationRepo = Depends(get_conversation_repo),
    messages: MessageRepo = Depends(get_message_repo),
):
    from app.services.conversation_service import get_system_stats

    return await get_system_stats(conversations, messages)


@router.get(
    "/{conversation_id}",
    response_model=ConversationOut,
    tags=["Conversations"],
    summary="Get a single conversation by ID",
    responses={
        **DB_ERROR_RESPONSES,
        404: {
            "model": ErrorResponse,
            "description": "Conversation not found (CONVERSATION_NOT_FOUND)",
        },
    },
)
async def get_conversation(
    conversation_id: str,
    repo: ConversationRepo = Depends(get_conversation_repo),
):
    return await conversation_service.get_conversation(repo, conversation_id)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationOut,
    tags=["Conversations"],
    summary="Rename or archive/unarchive a conversation",
    responses={
        **DB_ERROR_RESPONSES,
        404: {
            "model": ErrorResponse,
            "description": "Conversation not found (CONVERSATION_NOT_FOUND)",
        },
    },
)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    repo: ConversationRepo = Depends(get_conversation_repo),
):
    return await conversation_service.update_conversation(
        repo, conversation_id, request.title, request.archived
    )


@router.delete(
    "/{conversation_id}",
    response_model=DeleteResponse,
    tags=["Conversations"],
    summary="Delete a conversation and all its messages",
    responses={
        **DB_ERROR_RESPONSES,
        404: {
            "model": ErrorResponse,
            "description": "Conversation not found (CONVERSATION_NOT_FOUND)",
        },
    },
)
async def delete_conversation(
    conversation_id: str,
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo),
):
    return await conversation_service.delete_conversation(
        conv_repo, msg_repo, conversation_id
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=MessageList,
    tags=["Conversations"],
    summary="List messages in a conversation in ascending seq order",
    responses={
        **DB_ERROR_RESPONSES,
        404: {
            "model": ErrorResponse,
            "description": "Conversation not found (CONVERSATION_NOT_FOUND)",
        },
    },
)
async def list_messages(
    conversation_id: str,
    after_seq: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo),
):
    return await conversation_service.list_messages(
        conv_repo, msg_repo, conversation_id, after_seq, limit
    )


@router.post(
    "/{conversation_id}/chat",
    response_model=PersistedChatResponse,
    tags=["Persistent Chat"],
    summary="Send a persisted chat message",
    responses={
        **DB_ERROR_RESPONSES,
        404: {"model": ErrorResponse, "description": "Conversation not found"},
        409: {"model": ErrorResponse, "description": "Conversation archived or full"},
    },
)
async def persisted_chat(
    conversation_id: str,
    request: PersistedChatRequest,
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo),
):
    return await conversation_service.chat(
        conversation_id, request, conv_repo, msg_repo
    )


@router.post(
    "/{conversation_id}/messages/{message_id}/feedback",
    response_model=MessageOut,
    tags=["Conversations"],
    summary="Submit or replace feedback on an assistant message",
    responses={
        **DB_ERROR_RESPONSES,
        400: {
            "model": ErrorResponse,
            "description": "Feedback on invalid target (FEEDBACK_INVALID_TARGET)",
        },
        404: {
            "model": ErrorResponse,
            "description": "Conversation or message not found (CONVERSATION_NOT_FOUND / MESSAGE_NOT_FOUND)",
        },
    },
)
async def set_feedback(
    conversation_id: str,
    message_id: str,
    payload: FeedbackIn,
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo),
):
    return await conversation_service.set_message_feedback(
        conversation_id,
        message_id,
        payload.rating,
        payload.comment,
        conv_repo,
        msg_repo,
    )


@router.delete(
    "/{conversation_id}/messages/{message_id}/feedback",
    response_model=MessageOut,
    tags=["Conversations"],
    summary="Delete feedback on an assistant message",
    responses={
        **DB_ERROR_RESPONSES,
        400: {
            "model": ErrorResponse,
            "description": "Feedback on invalid target (FEEDBACK_INVALID_TARGET)",
        },
        404: {
            "model": ErrorResponse,
            "description": "Conversation or message not found (CONVERSATION_NOT_FOUND / MESSAGE_NOT_FOUND)",
        },
    },
)
async def delete_feedback(
    conversation_id: str,
    message_id: str,
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo),
):
    return await conversation_service.delete_message_feedback(
        conversation_id, message_id, conv_repo, msg_repo
    )
