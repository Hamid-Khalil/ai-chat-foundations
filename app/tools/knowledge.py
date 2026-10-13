"""Knowledge Base Search Tool."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.services.retrieval_service import get_retrieval_service


class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="The query to search the company knowledge base.",
    )
    top_k: int = Field(
        default=4, ge=1, le=6, description="Number of chunks to retrieve."
    )


@tool(args_schema=KnowledgeSearchInput)
async def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """Use this tool for questions about the company handbook, IT policy, product FAQ or onboarding documents.
    Do NOT use for general knowledge, the time, or arithmetic.
    Example query: 'How many paid leave days for contractors?'
    """
    service = get_retrieval_service()
    raw_chunks = await service.retrieve(
        query=query, top_k=top_k, document_ids=None, min_score=None
    )
    passed_chunks = [c for c in raw_chunks if c.passed_threshold]

    if not passed_chunks:
        return "No relevant passages found in the knowledge base."

    results = []
    for i, chunk in enumerate(passed_chunks, start=1):
        results.append(
            f"[{i}] {chunk.filename} (chunk {chunk.chunk_index}): {chunk.text}"
        )
    return "\n\n".join(results)
