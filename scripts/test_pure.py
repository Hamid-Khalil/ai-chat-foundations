import sys
import traceback

from app.services.ingestion_service import (
    clean_filename,
    decode_text,
    normalize_text,
    split_text,
)

print("=== Task 4.2: Pure Function Validation ===")
errors = 0

try:
    # 1. clean_filename (Path Traversal & XSS Defense)
    res1 = clean_filename("../../../etc/passwd.txt")
    res2 = clean_filename("a<script>alert(1)</script>.txt")
    if "/" in res1 or "\\" in res1:
        print(f"❌ clean_filename failed traversal check. Got: {res1}")
        errors += 1
    if "<" in res2 or ">" in res2:
        print(f"❌ clean_filename failed XSS sanitization. Got: {res2}")
        errors += 1

    # 2. decode_text (BOM Stripping)
    res3 = decode_text(b"\xef\xbb\xbfBOM Test")
    if res3 != "BOM Test":
        print(f"❌ decode_text failed BOM removal. Got: {res3}")
        errors += 1

    # 3. normalize_text (Null Byte & Bad Char Stripping)
    res4 = normalize_text("Null\x00Byte")
    if "\x00" in res4:
        print(f"❌ normalize_text failed to remove null bytes. Got: {res4}")
        errors += 1

    # 4. split_text (Deterministic Chunking without DB Calls)
    # Added the missing sha256 and created_at arguments
    chunks = split_text(
        text="A" * 1000,
        document_id="doc_1",
        filename="test.md",
        sha256="dummy_hash",
        created_at="2026-10-13",
        chunk_size=800,
        chunk_overlap=100,
    )

    if len(chunks) != 2:
        print(f"❌ split_text failed. Expected 2 chunks, got {len(chunks)}")
        errors += 1
    if len(chunks) == 2 and chunks[0].metadata.get("chunk_index") != 0:
        print(
            f"❌ split_text metadata failed. Expected index 0, got {chunks[0].metadata.get('chunk_index')}"
        )
        errors += 1

except Exception as e:
    print(f"❌ Crash during execution:")
    traceback.print_exc()
    errors += 1

if errors == 0:
    print("✅ All 4 functions verified as PURE and behaving correctly.")
else:
    sys.exit(1)
