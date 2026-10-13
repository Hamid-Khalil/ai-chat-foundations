import pytest
from langchain_core.documents import Document

from app.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_retrieval_empty_store():
    svc = RetrievalService()
    results = await svc.retrieve("any query")
    assert results == []


@pytest.mark.asyncio
async def test_retrieval_ranking_and_filters():
    svc = RetrievalService()

    docs = [
        Document(
            page_content="exact matching text",
            metadata={"document_id": "doc1", "filename": "doc1.txt", "chunk_index": 0},
        ),
        Document(
            page_content="completely different irrelevant string here",
            metadata={"document_id": "doc1", "filename": "doc1.txt", "chunk_index": 1},
        ),
        Document(
            page_content="another separate document content",
            metadata={"document_id": "doc2", "filename": "doc2.txt", "chunk_index": 0},
        ),
        Document(
            page_content="third document content",
            metadata={"document_id": "doc3", "filename": "doc3.txt", "chunk_index": 0},
        ),
    ]
    await svc.store.aadd_documents(docs)

    # 1. Identical text scores highest (1.0), rank starts at 1, sorted best first
    results = await svc.retrieve("exact matching text", top_k=4)
    assert len(results) > 0
    assert results[0].text == "exact matching text"
    assert results[0].score == 1.0
    assert results[0].rank == 1
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1

    # 2. top_k is respected
    top_2 = await svc.retrieve("exact matching text", top_k=2)
    assert len(top_2) <= 2

    # 3. Document filter for a single id
    filtered_single = await svc.retrieve("exact matching text", document_ids=["doc2"])
    assert all(r.document_id == "doc2" for r in filtered_single)

    # 4. Document filter for multiple ids
    filtered_multi = await svc.retrieve(
        "exact matching text", document_ids=["doc1", "doc2"]
    )
    assert all(r.document_id in ["doc1", "doc2"] for r in filtered_multi)
    assert not any(r.document_id == "doc3" for r in filtered_multi)

    # 5. passed_threshold flags match min_score
    strict_res = await svc.retrieve("exact matching text", min_score=0.99)
    for r in strict_res:
        if r.score >= 0.99:
            assert r.passed_threshold is True
        else:
            assert r.passed_threshold is False
