import pytest

from app.core.errors import DocUnsupportedTypeError
from app.services.ingestion_service import (
    clean_filename,
    decode_text,
    normalize_text,
    split_text,
)


def test_clean_filename():
    # Path traversal stripped
    assert clean_filename("../../a/b.txt") == "b.txt"
    # Angle brackets / script tags stripped
    assert clean_filename("a<script>.txt") == "ascript.txt"
    # Control characters removed (\x00, \x1f, \t)
    assert clean_filename("doc\x00\x1f\tname.txt") == "docname.txt"
    # Empty defaults to document.txt
    assert clean_filename("") == "document.txt"
    # 150-character names cut to 100
    long_name = "a" * 150
    cleaned = clean_filename(long_name)
    assert len(cleaned) == 100
    assert cleaned == "a" * 100


def test_decode_text():
    # Valid UTF-8
    assert decode_text(b"Standard UTF-8 content") == "Standard UTF-8 content"
    # UTF-8 with BOM stripped
    assert decode_text(b"\xef\xbb\xbfBOM Test") == "BOM Test"
    # Invalid byte sequences raise DocUnsupportedTypeError
    with pytest.raises(DocUnsupportedTypeError):
        decode_text(b"\xff\xfe\x00\x00")


def test_normalize_text():
    # CRLF converted to LF, NUL bytes stripped, blank lines collapsed
    dirty = "Line 1\r\nLine 2\x00\n\n\n\nLine 3   "
    assert normalize_text(dirty) == "Line 1\nLine 2\n\nLine 3"

    # Markdown table left structurally intact
    table = "| Header 1 | Header 2 |\n| :--- | :--- |\n| Val 1 | Val 2 |"
    assert normalize_text(table) == table


def test_split_text():
    # 1. Empty text returns no chunks
    assert (
        split_text("", "d1", "f.txt", "h", "t", chunk_size=200, chunk_overlap=40) == []
    )
    assert (
        split_text(
            "   \n\n  ", "d1", "f.txt", "h", "t", chunk_size=200, chunk_overlap=40
        )
        == []
    )

    # 2. Chunk size constraints, indexing, metadata, and start_index verification
    text = "A" * 1000
    chunk_size = 400
    chunk_overlap = 50
    chunks = split_text(
        text,
        document_id="doc1",
        filename="f.txt",
        sha256="hash123",
        created_at="2026-10-15T00:00:00Z",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    assert len(chunks) >= 2

    for idx, c in enumerate(chunks):
        # No chunk exceeds max chunk size
        assert len(c.page_content) <= chunk_size
        # Sequential indexing 0, 1, 2...
        assert c.metadata["chunk_index"] == idx
        # Complete metadata schema
        assert c.metadata["document_id"] == "doc1"
        assert c.metadata["filename"] == "f.txt"
        assert c.metadata["sha256"] == "hash123"
        assert c.metadata["created_at"] == "2026-10-15T00:00:00Z"
        # start_index accurately locates the slice in original text
        start = c.metadata["start_index"]
        assert text[start : start + len(c.page_content)] == c.page_content

    # 3. Consecutive chunks overlap
    for i in range(len(chunks) - 1):
        c1_end = chunks[i].metadata["start_index"] + len(chunks[i].page_content)
        c2_start = chunks[i + 1].metadata["start_index"]
        assert c1_end > c2_start  # Verified range overlap
