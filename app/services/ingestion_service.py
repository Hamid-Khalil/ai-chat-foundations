"""Knowledge Base document ingestion and chunking logic."""

import hashlib
import io
import logging
import os
import re
import time
from datetime import datetime, timezone

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.errors import (
    DocDuplicateError,
    DocEmptyError,
    DocLimitReachedError,
    DocNotFoundError,
    DocTooLargeError,
    DocUnsupportedTypeError,
)
from app.core.vector_store import get_vector_store
from app.schemas.knowledge import DeleteResult, DocumentInfo, DocumentList

logger = logging.getLogger(__name__)


def clean_filename(filename: str) -> str:
    """Strip path traversal sequences, tags, control chars, and truncate to 100 chars."""
    if not filename:
        return "document.txt"
    base = os.path.basename(filename)
    base = base.replace("..", "").replace("/", "").replace("\\", "")
    safe = re.sub(r"[\x00-\x1f\t<>]", "", base)
    if not safe:
        return "document.txt"
    return safe[:100]


def decode_text(data: bytes) -> str:
    """Decode raw bytes strictly, stripping UTF-8 BOM, raising DocUnsupportedTypeError on invalid bytes."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise DocUnsupportedTypeError(
            "Invalid byte sequence; only valid UTF-8 is supported."
        )


def normalize_text(text: str) -> str:
    """Remove control characters and normalize newlines while preserving formatting."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    document_id: str,
    filename: str,
    sha256: str = "",
    created_at: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    page_starts: list[tuple[int, int]] | None = None,
) -> list[Document]:
    """Split text recursively by paragraphs and sentences with metadata."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.create_documents([text])

    start_idx = 0
    for i, c in enumerate(chunks):
        start_idx = text.find(c.page_content, start_idx)
        c.metadata["document_id"] = document_id
        c.metadata["filename"] = filename
        c.metadata["chunk_index"] = i
        c.metadata["sha256"] = sha256
        c.metadata["created_at"] = created_at
        c.metadata["start_index"] = start_idx
        c.metadata["characters"] = len(text)

        if page_starts:
            page_num = 1
            for pos, pnum in reversed(page_starts):
                if start_idx >= pos:
                    page_num = pnum
                    break
            c.metadata["page"] = page_num

    return chunks


class IngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = get_vector_store()

    async def ingest_document(self, filename: str, data: bytes) -> DocumentInfo:
        start_time = time.perf_counter()
        sanitized_filename = clean_filename(filename)
        ext = (
            sanitized_filename.lower().split(".")[-1]
            if "." in sanitized_filename
            else ""
        )

        if ext not in ("txt", "md", "pdf"):
            raise DocUnsupportedTypeError(
                "Only .txt, .md, and .pdf files are supported."
            )

        if len(data) > self.settings.rag_max_upload_bytes:
            raise DocTooLargeError()

        page_starts = None
        if ext == "pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(data))
                if reader.is_encrypted:
                    raise DocUnsupportedTypeError("Encrypted PDF.")
                num_pages = len(reader.pages)
            except DocUnsupportedTypeError:
                raise
            except Exception:
                raise DocUnsupportedTypeError("Invalid or corrupt PDF.")

            if num_pages > 10:
                raise DocTooLargeError("PDF exceeds 10 pages.")

            page_starts = []
            text_parts = []
            current_len = 0
            for i, page in enumerate(reader.pages):
                pt = page.extract_text() or ""
                pt_norm = normalize_text(pt)
                page_starts.append((current_len, i + 1))
                text_parts.append(pt_norm)
                current_len += len(pt_norm) + 2

            raw_text = "\n\n".join(text_parts)
            text = normalize_text(raw_text)
        else:
            raw_text = decode_text(data)
            text = normalize_text(raw_text)

        if not text:
            raise DocEmptyError()

        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = doc_hash[:16]

        existing = self.vector_store.get(where={"document_id": document_id})
        if existing and existing["ids"]:
            raise DocDuplicateError(document_id)

        listing = await self.list_documents()
        if len(listing.documents) >= self.settings.rag_max_documents:
            raise DocLimitReachedError(
                f"Knowledge base is at maximum capacity ({self.settings.rag_max_documents} documents)."
            )

        created_at = datetime.now(timezone.utc).isoformat()
        chunks = split_text(
            text=text,
            document_id=document_id,
            filename=sanitized_filename,
            sha256=doc_hash,
            created_at=created_at,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
            page_starts=page_starts,
        )

        batch_size = self.settings.embedding_batch_size
        added_ids: list[str] = []
        try:
            # Atomic Ingestion Rollback:
            # A half-ingested document is significantly worse than a failed upload because
            # orphaned chunks pollute vector search, create split contexts, and cause phantom
            # citations for documents that are not completely indexed.
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                ids = [f"{document_id}:{c.metadata['chunk_index']}" for c in batch]
                await self.vector_store.aadd_documents(documents=batch, ids=ids)
                added_ids.extend(ids)
        except Exception:
            if added_ids:
                await self.vector_store.adelete(ids=added_ids)
            raise

        duration = time.perf_counter() - start_time
        logger.info(
            "Document ingested successfully: document_id=%s, chunks=%d, duration=%.3fs",
            document_id,
            len(chunks),
            duration,
        )

        return DocumentInfo(
            document_id=document_id,
            filename=sanitized_filename,
            sha256=doc_hash,
            chunks=len(chunks),
            characters=len(text),
            created_at=created_at,
        )

    async def list_documents(self) -> DocumentList:
        data = self.vector_store.get()
        metadatas = data.get("metadatas") or []

        doc_map: dict[str, dict] = {}
        for meta in metadatas:
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "sha256": meta.get("sha256", ""),
                    "chunks": 0,
                    "characters": meta.get("characters", 0),
                    "created_at": meta.get("created_at", ""),
                }
            doc_map[doc_id]["chunks"] += 1

        docs = [
            DocumentInfo(
                document_id=v["document_id"],
                filename=v["filename"],
                sha256=v["sha256"],
                chunks=v["chunks"],
                characters=v["characters"],
                created_at=v["created_at"],
            )
            for v in doc_map.values()
        ]

        return DocumentList(
            documents=docs,
            total_documents=len(docs),
            total_chunks=sum(d.chunks for d in docs),
        )

    async def delete_document(self, document_id: str) -> DeleteResult:
        start_time = time.perf_counter()
        data = self.vector_store.get(where={"document_id": document_id})
        ids = data.get("ids") or []
        if not ids:
            raise DocNotFoundError(f"Document {document_id} not found.")

        await self.vector_store.adelete(ids=ids)
        duration = time.perf_counter() - start_time
        logger.info(
            "Document deleted successfully: document_id=%s, deleted_chunks=%d, duration=%.3fs",
            document_id,
            len(ids),
            duration,
        )
        return DeleteResult(document_id=document_id, deleted_chunks=len(ids))


def get_ingestion_service() -> IngestionService:
    return IngestionService()
