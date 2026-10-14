"""Endpoints for stateful chat sessions powered by JSON history."""

from fastapi import APIRouter, status

from app.schemas.chat import ChatMessage
from app.schemas.chat_session import (
    ChatSessionDetail,
    ChatSessionSummary,
    ChatTurnRequest,
    CreateChatResponse,
)
from app.schemas.rag import RagReply
from app.services.json_history_service import get_json_history_service
from app.services.rag_service import get_rag_service

router = APIRouter()


@router.post("", response_model=CreateChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat() -> CreateChatResponse:
    """Create a new chat conversation."""
    service = get_json_history_service()
    return await service.create_chat()


@router.get("", response_model=list[ChatSessionSummary], status_code=status.HTTP_200_OK)
async def list_chats() -> list[ChatSessionSummary]:
    """List all saved chat conversations for the sidebar."""
    service = get_json_history_service()
    return await service.list_chats()


@router.get(
    "/{chat_id}", response_model=ChatSessionDetail, status_code=status.HTTP_200_OK
)
async def get_chat_history(chat_id: str) -> ChatSessionDetail:
    """Load message history for an existing chat session."""
    service = get_json_history_service()
    return await service.get_chat(chat_id)


@router.post(
    "/{chat_id}/messages", response_model=RagReply, status_code=status.HTTP_200_OK
)
async def send_chat_message(chat_id: str, payload: ChatTurnRequest) -> RagReply:
    """Send a message inside an existing chat, incorporating prior turns."""
    history_svc = get_json_history_service()
    rag_svc = get_rag_service()

    prior_turns = await history_svc.get_context_window(chat_id, max_turns=6)
    full_history = prior_turns + [ChatMessage(role="user", content=payload.message)]

    reply = await rag_svc.answer_conversation(
        history=full_history,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        thinking_level=payload.thinking_level,
    )

    await history_svc.append_turn(
        chat_id=chat_id,
        user_text=payload.message,
        assistant_text=reply.reply,
        sources=reply.sources,
    )

    return reply


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str) -> None:
    """Delete a chat session and its history."""
    service = get_json_history_service()
    await service.delete_chat(chat_id)
