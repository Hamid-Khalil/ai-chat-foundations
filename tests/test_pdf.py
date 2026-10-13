import pytest

from app.core.errors import DocEmptyError, DocTooLargeError, DocUnsupportedTypeError
from app.services.ingestion_service import IngestionService


def _create_test_pdf(pages_text: list[str]) -> bytes:
    """Construct a minimal valid PDF document with text."""
    objects = []
    font_obj_idx = 3 + len(pages_text) * 2
    page_obj_indices = [3 + i * 2 for i in range(len(pages_text))]

    objects.append((1, "<< /Type /Catalog /Pages 2 0 R >>"))
    kids_str = " ".join(f"{idx} 0 R" for idx in page_obj_indices)
    objects.append(
        (2, f"<< /Type /Pages /Kids [{kids_str}] /Count {len(pages_text)} >>")
    )

    for i, text in enumerate(pages_text):
        page_idx = 3 + i * 2
        content_idx = page_idx + 1
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_data = f"BT /F1 12 Tf 50 750 Td ({safe_text}) Tj ET"
        stream_len = len(stream_data)
        content_obj = f"<< /Length {stream_len} >>\nstream\n{stream_data}\nendstream"
        page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_idx} 0 R /Resources << /Font << /F1 {font_obj_idx} 0 R >> >> >>"
        objects.append((page_idx, page_obj))
        objects.append((content_idx, content_obj))

    objects.append(
        (font_obj_idx, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    )
    objects.sort(key=lambda x: x[0])

    out = b"%PDF-1.4\n"
    offsets = {0: 0}
    for obj_num, body in objects:
        offsets[obj_num] = len(out)
        out += f"{obj_num} 0 obj\n{body}\nendobj\n".encode("latin1")

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin1")
    for i in range(1, len(objects) + 1):
        out += f"{offsets[i]:010d} 00000 n \n".encode("latin1")

    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode(
        "latin1"
    )
    return out


def _create_empty_pdf() -> bytes:
    """Construct a minimal valid PDF without any text content streams."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n185\n%%EOF"
    )


@pytest.mark.asyncio
async def test_pdf_valid_ingest():
    svc = IngestionService()
    data = _create_test_pdf(["Hello World from page 1", "Second page content"])
    res = await svc.ingest_document("test_manual.pdf", data)
    assert res.chunks > 0

    listing = await svc.list_documents()
    assert "test_manual.pdf" in {d.filename for d in listing.documents}


@pytest.mark.asyncio
async def test_pdf_too_many_pages():
    svc = IngestionService()
    data_11 = _create_test_pdf([f"Page content {i+1}" for i in range(11)])
    with pytest.raises(DocTooLargeError):
        await svc.ingest_document("11_pages.pdf", data_11)


@pytest.mark.asyncio
async def test_pdf_empty_no_text():
    svc = IngestionService()
    data_empty = _create_empty_pdf()
    with pytest.raises(DocEmptyError):
        await svc.ingest_document("empty.pdf", data_empty)


@pytest.mark.asyncio
async def test_pdf_corrupt():
    svc = IngestionService()
    with pytest.raises(DocUnsupportedTypeError):
        await svc.ingest_document("corrupt.pdf", b"This is not a PDF file at all")
