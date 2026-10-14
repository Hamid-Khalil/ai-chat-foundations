"""API v1 router aggregator."""

from fastapi import APIRouter

from app.api.v1 import assistant, chat, knowledge, rag, sessions

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat")
api_router.include_router(assistant.router, prefix="/assistant")
api_router.include_router(knowledge.router, prefix="/knowledge")
api_router.include_router(rag.router, prefix="/rag")
api_router.include_router(sessions.router, prefix="/chats", tags=["Chat Sessions"])
