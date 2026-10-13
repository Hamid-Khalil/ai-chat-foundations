"""Service for vector retrieval and score normalization."""

import logging
import time

from app.core.config import get_settings
from app.core.errors import from_llm_exception
from app.core.vector_store import get_vector_store
from app.schemas.knowledge import SearchResultItem

logger = logging.getLogger(__name__)


class RetrievalService:
    """Performs semantic search against persistent Chroma store."""

    def __init__(self) -> None:
        self.store = get_vector_store()
        self.settings = get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[SearchResultItem]:
        """Search Chroma, invert cosine distance to similarity, apply threshold.

        Score formula:
            Chroma returns cosine distance D in range [0, 2].
            Similarity S = round(1.0 - D, 4), clamped to [0.0, 1.0].
            Higher score = more relevant.
        """
        start_time = time.monotonic()
        k = top_k if top_k is not None else self.settings.rag_top_k_default
        k = min(k, self.settings.rag_top_k_max)
        threshold = min_score if min_score is not None else self.settings.rag_min_score

        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        try:
            raw_results = await self.store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=where_filter,
            )
        except Exception as exc:
            raise from_llm_exception(exc) from exc

        results: list[SearchResultItem] = []
        for doc, distance in raw_results:
            score = round(max(0.0, min(1.0, 1.0 - distance)), 4)
            passed = score >= threshold

            item = SearchResultItem(
                rank=0,
                document_id=doc.metadata.get("document_id", "unknown"),
                filename=doc.metadata.get("filename", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                score=score,
                passed_threshold=passed,
                text=doc.page_content,
                page=(
                    int(doc.metadata.get("page"))
                    if doc.metadata.get("page") is not None
                    else None
                ),
            )
            results.append(item)

        results.sort(key=lambda x: x.score, reverse=True)

        for idx, item in enumerate(results, start=1):
            item.rank = idx

        duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        passed_count = sum(1 for r in results if r.passed_threshold)
        logger.info(
            "Retrieval completed: k=%d returned=%d passed=%d duration_ms=%.2f",
            k,
            len(results),
            passed_count,
            duration_ms,
        )

        return results


def get_retrieval_service() -> RetrievalService:
    return RetrievalService()
