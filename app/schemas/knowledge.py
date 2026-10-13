"""Schemas for knowledge base management and search operations."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentInfo(BaseModel):
    """Metadata summary of an ingested document."""

    document_id: str = Field(description="Unique hex ID of the document.")
    filename: str = Field(description="Sanitized document filename.")
    sha256: str = Field(description="SHA-256 hash of the normalized content.")
    chunks: int = Field(description="Number of chunks stored for this document.")
    characters: int = Field(description="Total character count of normalized content.")
    created_at: str = Field(description="ISO 8601 UTC timestamp.")


class DocumentList(BaseModel):
    """List of all indexed documents in the knowledge base."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteResult(BaseModel):
    """Result of document deletion."""

    document_id: str
    deleted_chunks: int


class SearchRequest(BaseModel):
    """Payload for POST /knowledge/search."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How many days of paid leave do full-time employees receive?",
                "top_k": 4,
                "document_ids": None,
                "min_score": 0.66,
            }
        }
    )

    query: str = Field(
        ..., min_length=1, max_length=500, description="Search query string."
    )
    top_k: int = Field(
        default=4, ge=1, le=8, description="Number of results to retrieve."
    )
    document_ids: list[str] | None = Field(
        default=None, description="Optional document filter."
    )
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Score threshold override."
    )


class SearchResultItem(BaseModel):
    """Single search match item."""

    rank: int
    document_id: str
    filename: str
    chunk_index: int
    score: float
    passed_threshold: bool
    text: str
    page: int | None = Field(
        default=None, description="Page number if document is a PDF"
    )


class SearchResponse(BaseModel):
    """Response returned by POST /knowledge/search."""

    query: str
    top_k: int
    min_score: float
    results: list[SearchResultItem]
