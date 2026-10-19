from fastapi import APIRouter, Depends, Query, status
from typing import Optional
from app.schemas.conversations import (
    ConversationOut, ConversationList, ConversationCreate, 
    ConversationUpdate, MessageList, DeleteResponse
)
from app.api.deps import get_conversation_repo, get_message_repo
from app.repositories.protocols import ConversationRepo, MessageRepo
from app.services import conversation_service

router = APIRouter()

@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED, tags=["Conversations"], summary="Create Conversation")
async def create_conversation(
    request: Optional[ConversationCreate] = None,
    repo: ConversationRepo = Depends(get_conversation_repo)
):
    title = request.title if request else None
    return await conversation_service.create_conversation(repo, title)

@router.get("", response_model=ConversationList, tags=["Conversations"], summary="List Conversations")
async def list_conversations(
    archived: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    repo: ConversationRepo = Depends(get_conversation_repo)
):
    return await conversation_service.list_conversations(repo, archived, limit, cursor)

@router.get("/{conversation_id}", response_model=ConversationOut, tags=["Conversations"], summary="Get Conversation")
async def get_conversation(
    conversation_id: str,
    repo: ConversationRepo = Depends(get_conversation_repo)
):
    return await conversation_service.get_conversation(repo, conversation_id)

@router.patch("/{conversation_id}", response_model=ConversationOut, tags=["Conversations"], summary="Update Conversation")
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    repo: ConversationRepo = Depends(get_conversation_repo)
):
    return await conversation_service.update_conversation(repo, conversation_id, request.title, request.archived)

@router.delete("/{conversation_id}", response_model=DeleteResponse, tags=["Conversations"], summary="Delete Conversation")
async def delete_conversation(
    conversation_id: str,
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo)
):
    return await conversation_service.delete_conversation(conv_repo, msg_repo, conversation_id)

@router.get("/{conversation_id}/messages", response_model=MessageList, tags=["Conversations"], summary="List Messages")
async def list_messages(
    conversation_id: str,
    after_seq: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    conv_repo: ConversationRepo = Depends(get_conversation_repo),
    msg_repo: MessageRepo = Depends(get_message_repo)
):
    return await conversation_service.list_messages(conv_repo, msg_repo, conversation_id, after_seq, limit)
