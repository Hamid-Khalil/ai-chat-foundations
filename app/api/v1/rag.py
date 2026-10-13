"""RAG Chat endpoints."""

from fastapi import APIRouter, status

from app.schemas.rag import RagAskRequest, RagMessagesRequest, RagReply
from app.services.rag_service import get_rag_service

router = APIRouter()

COMMON_RAG_RESPONSES = {
    422: {
        "description": "Request validation error (e.g. prompt length, alternating sequence)."
    },
    429: {"description": "LLM provider rate limit exceeded."},
    502: {"description": "LLM provider upstream error."},
    504: {"description": "LLM provider request timed out."},
}


@router.post(
    "/ask",
    response_model=RagReply,
    tags=["RAG Chat"],
    summary="Ask a single question grounded entirely in the knowledge base",
    description="Retrieves top chunks from the knowledge base, enforces similarity thresholding, builds an isolated context block, and synthesizes an answer with inline citations. Short-circuits with zero tokens when no chunk passes threshold.",
    status_code=status.HTTP_200_OK,
    responses=COMMON_RAG_RESPONSES,
)
async def ask_question(payload: RagAskRequest) -> RagReply:
    service = get_rag_service()
    return await service.answer_question(
        question=payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        thinking_level=payload.thinking_level,
    )


@router.post(
    "/messages",
    response_model=RagReply,
    tags=["RAG Chat"],
    summary="Ask a follow-up question with conversation history",
    description="Condenses conversation history into a standalone search query, fetches context chunks via semantic retrieval, and synthesizes a grounded response citing numbered sources.",
    status_code=status.HTTP_200_OK,
    responses=COMMON_RAG_RESPONSES,
)
async def ask_messages(payload: RagMessagesRequest) -> RagReply:
    service = get_rag_service()
    return await service.answer_conversation(
        history=payload.messages,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        thinking_level=payload.thinking_level or "low",
    )
