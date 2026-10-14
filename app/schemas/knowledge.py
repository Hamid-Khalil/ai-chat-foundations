"""Schemas for knowledge base management and search operations."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


def validate_document_ids_list(v: list[str] | None) -> list[str] | None:
    """Validate that document_ids, if provided, has 1-20 items, each 1-64 chars."""
    if v is None:
        return None
    if len(v) < 1 or len(v) > 20:
        raise ValueError("document_ids must contain between 1 and 20 items.")
    for doc_id in v:
        if not isinstance(doc_id, str) or not doc_id.strip() or len(doc_id) > 64:
            raise ValueError(
                "Each document_id must be a non-empty string of at most 64 characters."
            )
    return v


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
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": "How many days of paid leave do full-time employees receive?",
                "top_k": 4,
                "document_ids": None,
                "min_score": 0.66,
            }
        },
    )

    query: str = Field(
        ..., min_length=1, max_length=500, description="Search query string."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=8, description="Number of results to retrieve."
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional document filter (1-20 ids, each <= 64 chars).",
    )
    min_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Score threshold override."
    )

    @field_validator("document_ids")
    @classmethod
    def _check_document_ids(cls, v: list[str] | None) -> list[str] | None:
        return validate_document_ids_list(v)


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
