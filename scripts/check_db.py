import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.mongo import close_mongo_client, get_database, init_mongo_client


async def main():
    await init_mongo_client()
    db = get_database()

    conv_count = await db.conversations.count_documents({})
    msg_count = await db.messages.count_documents({})
    doc_count = await db.documents.count_documents({})

    print("--- Database Counts ---")
    print(f"Conversations: {conv_count}")
    print(f"Messages:      {msg_count}")
    print(f"Documents:     {doc_count}\n")

    # Fetch all records
    convs = await db.conversations.find({}).to_list(length=None)
    conv_ids = {c["_id"]: c.get("message_count", 0) for c in convs}
    msgs = await db.messages.find({}).to_list(length=None)

    orphans = 0
    mismatch_convs = 0
    duplicate_seqs = 0
    gaps = 0

    msgs_by_conv = {}
    for m in msgs:
        cid = m["conversation_id"]
        msgs_by_conv.setdefault(cid, []).append(m["seq"])
        if cid not in conv_ids:
            orphans += 1
            print(
                f"Orphan message found: msg_id={m['_id']} references missing conv_id={cid}"
            )

    for cid, expected_count in conv_ids.items():
        actual_seqs = msgs_by_conv.get(cid, [])
        actual_count = len(actual_seqs)

        if actual_count != expected_count:
            mismatch_convs += 1
            print(
                f"Mismatch in {cid}: DB message_count={expected_count}, but actual chunks={actual_count}"
            )

        if actual_count > 0:
            actual_seqs.sort()
            if len(set(actual_seqs)) != actual_count:
                duplicate_seqs += 1
                print(f"Duplicate seqs in {cid}: {actual_seqs}")

            # Gaps: The highest sequence number should ideally equal the count
            if actual_seqs[-1] != actual_count:
                gaps += 1

    print("--- Integrity Report ---")
    print(f"(a) Orphans: {orphans}")
    print(f"(b) Mismatched message counts: {mismatch_convs}")
    print(f"(c) Sequence gaps: {gaps} (informational - gaps are allowed)")
    print(f"(d) Duplicate seqs: {duplicate_seqs}")

    await close_mongo_client()

    if orphans > 0 or mismatch_convs > 0 or duplicate_seqs > 0:
        sys.exit(1)

    print("\nIntegrity check PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
