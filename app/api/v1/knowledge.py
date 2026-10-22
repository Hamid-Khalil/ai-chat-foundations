"""Knowledge Base document management and search endpoints."""

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.deps import get_document_repo
from app.core.config import get_settings
from app.repositories.protocols import DocumentRepo
from app.schemas.knowledge import (
    DeleteResult,
    DocumentInfo,
    DocumentList,
    SearchRequest,
    SearchResponse,
)
from app.services.ingestion_service import get_ingestion_service
from app.services.retrieval_service import get_retrieval_service

router = APIRouter()


@router.post(
    "/documents",
    response_model=DocumentInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["Knowledge Base"],
    summary="Upload and ingest a document into the knowledge base",
    description="Accepts a .txt, .md, or .pdf file, strips traversal sequences, normalizes formatting, segments content into overlapping chunks, generates embeddings, and persists them into Chroma and the MongoDB registry.",
    responses={
        400: {
            "description": "Document is empty or contains only whitespace (DOC_EMPTY)."
        },
        409: {
            "description": "Identical document already exists (DOC_DUPLICATE) or document limit reached (DOC_LIMIT_REACHED)."
        },
        413: {
            "description": "Document exceeds maximum upload size or 10-page PDF limit (DOC_TOO_LARGE)."
        },
        415: {
            "description": "Unsupported file format or corrupt/non-UTF-8 content (DOC_UNSUPPORTED_TYPE)."
        },
        422: {"description": "Request validation error."},
        429: {"description": "Upstream embedding provider rate limit reached."},
        502: {"description": "Embedding provider upstream communication failure."},
        503: {"description": "Database unavailable (DB_UNAVAILABLE)."},
        504: {"description": "Embedding provider request timed out."},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    repo: DocumentRepo = Depends(get_document_repo),
) -> DocumentInfo:
    """Read upload stream with size cap and delegate ingestion to IngestionService with registry."""
    settings = get_settings()
    filename = file.filename or ""
    data = await file.read(settings.rag_max_upload_bytes + 1)
    service = get_ingestion_service()
    return await service.ingest_document(filename=filename, data=data, repo=repo)


@router.get(
    "/documents",
    response_model=DocumentList,
    tags=["Knowledge Base"],
    summary="List all indexed documents in the knowledge base",
    description="Returns an aggregated listing of all documents stored in the MongoDB registry with their chunk counts.",
    status_code=status.HTTP_200_OK,
)
async def list_documents(
    repo: DocumentRepo = Depends(get_document_repo),
) -> DocumentList:
    """List documents directly from the MongoDB registry."""
    service = get_ingestion_service()
    return await service.list_documents(repo=repo)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResult,
    tags=["Knowledge Base"],
    summary="Delete a document and all its chunks",
    description="Permanently deletes chunks from Chroma first, then removes the registry record.",
    status_code=status.HTTP_200_OK,
    responses={
        404: {
            "description": "Document ID not found in knowledge base (DOC_NOT_FOUND)."
        },
    },
)
async def delete_document(
    document_id: str,
    repo: DocumentRepo = Depends(get_document_repo),
) -> DeleteResult:
    """Remove chunks from Chroma then remove registry record."""
    service = get_ingestion_service()
    return await service.delete_document(document_id, repo=repo)


# Architectural Note: Search and RAG endpoints do NOT touch MongoDB.
# They continue operating purely against Chroma and the LLM when MongoDB is down.
@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
    summary="Semantic search over indexed chunks without LLM synthesis",
    description="Embeds the input query and executes vector similarity search against Chroma.",
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error on search parameters."},
        429: {"description": "Embedding provider rate limit exceeded."},
        502: {"description": "Embedding provider upstream error."},
        504: {"description": "Embedding provider request timed out."},
    },
)
async def search_knowledge_base(payload: SearchRequest) -> SearchResponse:
    """Search the vector store directly without touching MongoDB."""
    settings = get_settings()
    effective_top_k = (
        payload.top_k if payload.top_k is not None else settings.rag_top_k_default
    )
    effective_min_score = (
        payload.min_score if payload.min_score is not None else settings.rag_min_score
    )

    service = get_retrieval_service()
    results = await service.retrieve(
        query=payload.query.strip(),
        top_k=effective_top_k,
        document_ids=payload.document_ids,
        min_score=effective_min_score,
    )
    return SearchResponse(
        query=payload.query.strip(),
        top_k=effective_top_k,
        min_score=effective_min_score,
        results=results,
    )
