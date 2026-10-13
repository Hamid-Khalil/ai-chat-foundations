"""Knowledge Base document management and search endpoints."""

from fastapi import APIRouter, File, UploadFile, status

from app.core.config import get_settings
from app.core.errors import DocTooLargeError, DocUnsupportedTypeError
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
    description="Accepts a .txt or .md file, strips traversal sequences, normalizes formatting, segments content into overlapping chunks, generates embeddings, and persists them into Chroma.",
    responses={
        400: {
            "description": "Document is empty or contains only whitespace (DOC_EMPTY)."
        },
        409: {
            "description": "Identical document already exists in the knowledge base (DOC_DUPLICATE)."
        },
        413: {"description": "Document exceeds maximum upload size (DOC_TOO_LARGE)."},
        415: {
            "description": "Unsupported file format. Only .txt and .md are accepted (DOC_UNSUPPORTED_TYPE)."
        },
        422: {"description": "Request validation error."},
        429: {"description": "Upstream embedding provider rate limit reached."},
        502: {"description": "Embedding provider upstream communication failure."},
        504: {"description": "Embedding provider request timed out."},
    },
)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Accept .txt/.md upload, validate size, and delegate to IngestionService."""
    settings = get_settings()
    filename = file.filename or ""

    if not (filename.lower().endswith(".txt") or filename.lower().endswith(".md")):
        raise DocUnsupportedTypeError("Only .txt and .md files are supported.")

    data = await file.read(settings.rag_max_upload_bytes + 1)
    if len(data) > settings.rag_max_upload_bytes:
        raise DocTooLargeError()

    service = get_ingestion_service()
    return await service.ingest_document(filename=filename, data=data)


@router.get(
    "/documents",
    response_model=DocumentList,
    tags=["Knowledge Base"],
    summary="List all indexed documents in the knowledge base",
    description="Returns an aggregated listing of all documents stored in Chroma along with chunk counts, character lengths, and cryptographic SHA-256 hashes.",
    status_code=status.HTTP_200_OK,
)
async def list_documents() -> DocumentList:
    """Aggregate indexed chunks by document."""
    service = get_ingestion_service()
    return await service.list_documents()


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResult,
    tags=["Knowledge Base"],
    summary="Delete a document and all its chunks",
    description="Permanently deletes a document and all of its associated vector chunks from the persistent Chroma store.",
    status_code=status.HTTP_200_OK,
    responses={
        404: {
            "description": "Document ID not found in knowledge base (DOC_NOT_FOUND)."
        },
    },
)
async def delete_document(document_id: str) -> DeleteResult:
    """Remove all chunks associated with a document_id."""
    service = get_ingestion_service()
    return await service.delete_document(document_id)


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
    summary="Semantic search over indexed chunks without LLM synthesis",
    description="Embeds the input query and executes vector similarity search against Chroma, normalizing distance metrics to similarity scores in range [0.0, 1.0].",
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error on search parameters."},
        429: {"description": "Embedding provider rate limit exceeded."},
        502: {"description": "Embedding provider upstream error."},
        504: {"description": "Embedding provider request timed out."},
    },
)
async def search_knowledge_base(payload: SearchRequest) -> SearchResponse:
    """Search the vector store and return relevant chunks with similarity scores."""
    service = get_retrieval_service()
    results = await service.retrieve(
        query=payload.query.strip(),
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        min_score=payload.min_score,
    )
    return SearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        results=results,
    )
