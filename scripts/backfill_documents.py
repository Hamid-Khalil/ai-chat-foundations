import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.mongo import close_mongo_client, get_database, init_mongo_client
from app.core.vector_store import get_vector_store
from app.repositories.mongo_documents import MongoDocumentRepo


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill MongoDB Document Registry from Chroma chunks"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without modifying the registry",
    )
    args = parser.parse_args()

    await init_mongo_client()
    db = get_database()
    repo = MongoDocumentRepo(db.documents)
    vs = get_vector_store()

    # 1. Fetch chunk metadata and content from the underlying Chroma collection
    all_data = vs._collection.get(include=["metadatas", "documents"])
    metadatas = all_data.get("metadatas", []) or []
    texts = all_data.get("documents", []) or []

    chroma_docs = {}
    for meta, text in zip(metadatas, texts):
        if not meta or "document_id" not in meta:
            continue
        doc_id = meta["document_id"]
        if doc_id not in chroma_docs:
            chroma_docs[doc_id] = {
                "id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "sha256": meta.get("sha256", ""),
                "created_at": meta.get("created_at"),
                "chunks": 0,
                "characters": 0,
            }
        chroma_docs[doc_id]["chunks"] += 1
        chroma_docs[doc_id]["characters"] += len(text or "")

    # 2. Fetch existing MongoDB registry records
    registry_docs = await repo.list()
    registry_by_id = {d["id"]: d for d in registry_docs}

    # 3. Detect drift in both directions
    missing_in_registry = [
        d for did, d in chroma_docs.items() if did not in registry_by_id
    ]
    orphan_in_registry = [
        d for did, d in registry_by_id.items() if did not in chroma_docs
    ]

    if args.check:
        print("Drift Check:")
        print(f"- In Chroma without registry record: {len(missing_in_registry)}")
        print(f"- In registry without Chroma chunks: {len(orphan_in_registry)}")
        await close_mongo_client()
        return

    # 4. Perform backfill
    added_count = 0
    for doc in missing_in_registry:
        await repo.insert(doc)
        added_count += 1

    print(f"Backfill complete: {added_count} documents added to registry.")
    await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
