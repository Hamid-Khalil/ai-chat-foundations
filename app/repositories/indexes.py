from pymongo import ASCENDING, DESCENDING


async def ensure_indexes(db):
    # 1. conversations: (archived, updated_at desc, _id desc)
    await db.conversations.create_index(
        [("archived", ASCENDING), ("updated_at", DESCENDING), ("_id", DESCENDING)],
        name="idx_conversation_list",
    )

    # 2. messages: unique (conversation_id, seq)
    await db.messages.create_index(
        [("conversation_id", ASCENDING), ("seq", ASCENDING)],
        unique=True,
        name="idx_message_conv_seq",
    )

    # 3. documents: unique (sha256)
    await db.documents.create_index(
        [("sha256", ASCENDING)], unique=True, name="idx_documents_sha256"
    )
