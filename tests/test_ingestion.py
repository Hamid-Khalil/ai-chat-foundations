from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.errors import (
    DocDuplicateError,
    DocEmptyError,
    DocLimitReachedError,
    DocNotFoundError,
    DocTooLargeError,
    DocUnsupportedTypeError,
)
from app.services.ingestion_service import IngestionService
from tests.conftest import FailingEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


@pytest.mark.asyncio
async def test_ingest_document():
    svc = IngestionService()
    res = await svc.ingest_document("test.txt", b"Hello world")
    assert res.chunks > 0
    assert res.filename == "test.txt"


@pytest.mark.asyncio
async def test_ingest_duplicate():
    svc = IngestionService()
    await svc.ingest_document("test.txt", b"Hello world")
    with pytest.raises(DocDuplicateError):
        await svc.ingest_document("duplicate.txt", b"Hello world")


@pytest.mark.asyncio
async def test_ingest_unsupported():
    svc = IngestionService()
    # .pdf is now supported due to Task 4.4; testing truly unsupported format .exe
    with pytest.raises(DocUnsupportedTypeError):
        await svc.ingest_document("test.exe", b"MZ\x90\x00BinaryData")


@pytest.mark.asyncio
async def test_ingest_empty():
    svc = IngestionService()
    with pytest.raises(DocEmptyError):
        await svc.ingest_document("test.txt", b"   \n  ")


@pytest.mark.asyncio
async def test_ingest_too_large():
    svc = IngestionService()
    orig = svc.settings.rag_max_upload_bytes
    try:
        svc.settings.rag_max_upload_bytes = 10
        with pytest.raises(DocTooLargeError):
            await svc.ingest_document("large.txt", b"This text exceeds ten bytes.")
    finally:
        svc.settings.rag_max_upload_bytes = orig


@pytest.mark.asyncio
async def test_ingest_limit_reached():
    svc = IngestionService()
    orig = svc.settings.rag_max_documents
    try:
        svc.settings.rag_max_documents = 1
        await svc.ingest_document("doc1.txt", b"Content 1")
        with pytest.raises(DocLimitReachedError):
            await svc.ingest_document("doc2.txt", b"Content 2")
    finally:
        svc.settings.rag_max_documents = orig


@pytest.mark.asyncio
async def test_ingest_rollback(tmp_path):
    failing_store = Chroma(
        collection_name="rollback_collection",
        embedding_function=FailingEmbeddings(),
        persist_directory=str(tmp_path / "chroma_rollback"),
    )
    with patch(
        "app.services.ingestion_service.get_vector_store",
        return_value=failing_store,
    ):
        svc = IngestionService()
        svc.settings.embedding_batch_size = 1
        svc.settings.rag_chunk_size = 500

        text = "A" * 1200  # Generates 3 chunks, fails on batch 2
        with pytest.raises(Exception):
            await svc.ingest_document("fail.txt", text.encode("utf-8"))

        data = failing_store.get()
        assert len(data.get("ids", [])) == 0


@pytest.mark.asyncio
async def test_list_and_delete_documents():
    svc = IngestionService()
    doc1 = await svc.ingest_document("doc1.txt", b"Content for document one")
    doc2 = await svc.ingest_document("doc2.txt", b"Content for document two")

    # 1. list_documents groups correctly
    listing = await svc.list_documents()
    assert listing.total_documents >= 2
    filenames = {d.filename for d in listing.documents}
    assert "doc1.txt" in filenames
    assert "doc2.txt" in filenames

    # 2. delete returns right count and chunks are purged
    del_res = await svc.delete_document(doc1.document_id)
    assert del_res.document_id == doc1.document_id
    assert del_res.deleted_chunks == doc1.chunks

    # Verify doc1 is gone
    after_del = await svc.list_documents()
    remaining_ids = {d.document_id for d in after_del.documents}
    assert doc1.document_id not in remaining_ids

    # 3. delete of an unknown id raises DocNotFoundError
    with pytest.raises(DocNotFoundError):
        await svc.delete_document("non_existent_doc_id_999")
