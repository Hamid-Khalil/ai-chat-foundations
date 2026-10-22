from typing import Optional

from app.repositories.protocols import DocumentRepo

"""Knowledge Base document ingestion and chunking logic."""

import hashlib
import io
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.errors import (
    AppError,
    DocDuplicateError,
    DocEmptyError,
    DocLimitReachedError,
    DocNotFoundError,
    DocTooLargeError,
    DocUnsupportedTypeError,
    from_llm_exception,
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
    except UnicodeDecodeError as exc:
        raise DocUnsupportedTypeError(
            "Invalid byte sequence; only valid UTF-8 is supported."
        ) from exc


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
        add_start_index=True,
    )
    doc = Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "filename": filename,
            "sha256": sha256,
            "created_at": created_at,
            "characters": len(text),
        },
    )
    chunks = splitter.split_documents([doc])

    search_pos = 0
    for i, c in enumerate(chunks):
        start_idx = c.metadata.get("start_index", -1)
        if start_idx is None or start_idx < 0:
            start_idx = text.find(c.page_content, search_pos)
        search_pos = max(0, start_idx)

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

    async def ingest_document(
        self, filename: str, data: bytes, repo: Optional[DocumentRepo] = None
    ) -> DocumentInfo:
        start_time = time.perf_counter()
        sanitized_filename = clean_filename(filename)
        ext = (
            sanitized_filename.lower().split(".")[-1]
            if "." in sanitized_filename
            else ""
        )

        # 1. Check extension (.txt, .md, and .pdf for Task 4.4)
        if ext not in ("txt", "md", "pdf"):
            raise DocUnsupportedTypeError(
                "Only .txt, .md, and .pdf files are supported."
            )

        # 2. Check byte size limit
        if len(data) > self.settings.rag_max_upload_bytes:
            raise DocTooLargeError()

        # 3. Decode and normalize text
        page_starts = None
        if ext == "pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(data))
                if reader.is_encrypted:
                    raise DocUnsupportedTypeError("Encrypted PDF.")
                num_pages = len(reader.pages)
            except DocUnsupportedTypeError:
                raise
            except Exception as exc:
                raise DocUnsupportedTypeError("Invalid or corrupt PDF.") from exc

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

        # 4 & 5. Pre-check duplicate and capacity limit before spending embedding quota
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if repo is not None:
            # (1) Read registry for duplicate (sha256) and count(): DB outage returns 503 before any embeddings
            existing = await repo.get_by_sha256(doc_hash)
            if existing:
                raise DocDuplicateError(existing.get("id", doc_hash[:16]))
            doc_count = await repo.count()
            if doc_count >= self.settings.rag_max_documents:
                raise DocLimitReachedError(
                    f"Knowledge base is at maximum capacity ({self.settings.rag_max_documents} documents)."
                )
        else:
            existing = self.vector_store.get(where={"sha256": doc_hash})
            if existing and existing.get("ids"):
                metadatas = existing.get("metadatas") or []
                existing_doc_id = (
                    metadatas[0].get("document_id", doc_hash[:16])
                    if metadatas and metadatas[0]
                    else doc_hash[:16]
                )
                raise DocDuplicateError(existing_doc_id)

            listing = await self.list_documents()
            if len(listing.documents) >= self.settings.rag_max_documents:
                raise DocLimitReachedError(
                    f"Knowledge base is at maximum capacity ({self.settings.rag_max_documents} documents)."
                )

        # 6. Generate random hex document_id and split into chunks
        document_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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

        # 7 & 8. Store in batches with atomic rollback and LLM exception mapping
        batch_size = self.settings.embedding_batch_size
        added_ids: list[str] = []
        try:
            # Atomic Ingestion Rollback:
            # A half-ingested document is worse than a failed upload because orphaned chunks
            # pollute future search results, distort rankings, and produce incomplete answers.
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                ids = [f"{document_id}:{c.metadata['chunk_index']}" for c in batch]
                await self.vector_store.aadd_documents(documents=batch, ids=ids)
                added_ids.extend(ids)
        except Exception as exc:
            if added_ids:
                await self.vector_store.adelete(ids=added_ids)
            if isinstance(exc, AppError):
                raise
            raise from_llm_exception(exc) from exc

        # (3) Only then insert the registry record into MongoDB
        if repo is not None:
            registry_doc = {
                "id": document_id,
                "filename": sanitized_filename,
                "sha256": doc_hash,
                "chunks": len(chunks),
                "characters": len(text),
                "created_at": created_at,
            }
            try:
                await repo.insert(registry_doc)
            except Exception as exc:
                # If step 3 fails: delete the chunks just stored in Chroma and re-raise
                if added_ids:
                    try:
                        await self.vector_store.adelete(ids=added_ids)
                    except Exception:
                        pass
                exc_str = str(exc).lower()
                if (
                    "duplicate" in exc_str
                    or "e11000" in exc_str
                    or getattr(exc, "code", None) == 11000
                ):
                    raise DocDuplicateError(document_id)
                raise

        duration = time.perf_counter() - start_time
        logger.info(
            "Document ingested: document_id=%s chunks=%d duration=%.3fs",
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

    async def list_documents(self, repo: Optional[DocumentRepo] = None) -> DocumentList:
        if repo is not None:
            # Served from MongoDB registry
            records = await repo.list()
            docs = [
                DocumentInfo(
                    document_id=r["id"],
                    filename=r.get("filename", "unknown"),
                    sha256=r.get("sha256", ""),
                    chunks=(
                        r.get("chunks")
                        if r.get("chunks") is not None
                        else r.get("chunks", 0)
                    ),
                    characters=(
                        r.get("characters")
                        if r.get("characters") is not None
                        else r.get("characters", 0)
                    ),
                    created_at=str(r.get("created_at", "")),
                )
                for r in records
            ]
            return DocumentList(
                documents=docs,
                total_documents=len(docs),
                total_chunks=sum(d.chunks for d in docs),
            )

        # Fallback to scanning Chroma chunks if called without repo
        data = self.vector_store.get()
        metadatas = data.get("metadatas") or []

        doc_map: dict[str, dict] = {}
        for meta in metadatas:
            if not meta:
                continue
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

    async def delete_document(
        self, document_id: str, repo: Optional[DocumentRepo] = None
    ) -> DeleteResult:
        start_time = time.perf_counter()
        if repo is not None:
            # 1. Look up document in registry first (DOC_NOT_FOUND if missing)
            doc = await repo.get(document_id)
            if not doc:
                raise DocNotFoundError(f"Document {document_id} not found.")

            # 2. Delete order: delete its chunks from Chroma first, then the registry record.
            # Architectural reason: if the second step fails you are left with a registered document
            # that has no chunks (visible and harmless, and a retry works) — the other order
            # could leave chunks that still produce answers from a 'deleted' document
            # (invisible and harmful). A retry of a half-finished delete must succeed.
            data = self.vector_store.get(where={"document_id": document_id})
            ids = data.get("ids") or []
            if ids:
                await self.vector_store.adelete(ids=ids)

            # 3. Delete from registry
            await repo.delete(document_id)
            duration = time.perf_counter() - start_time
            logger.info(
                "Document deleted: document_id=%s deleted_chunks=%d duration=%.3fs",
                document_id,
                len(ids),
                duration,
            )
            return DeleteResult(document_id=document_id, deleted_chunks=len(ids))

        # Fallback when called without repo
        data = self.vector_store.get(where={"document_id": document_id})
        ids = data.get("ids") or []
        if not ids:
            raise DocNotFoundError(f"Document {document_id} not found.")

        await self.vector_store.adelete(ids=ids)
        duration = time.perf_counter() - start_time
        logger.info(
            "Document deleted: document_id=%s deleted_chunks=%d duration=%.3fs",
            document_id,
            len(ids),
            duration,
        )
        return DeleteResult(document_id=document_id, deleted_chunks=len(ids))


def get_ingestion_service() -> IngestionService:
    return IngestionService()
