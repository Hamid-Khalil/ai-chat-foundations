from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.database import AsyncDatabase
import logging

logger = logging.getLogger(__name__)

async def ensure_indexes(db: AsyncDatabase) -> None:
    """Create explicit unique and list-serving indexes for collections."""
    try:
        await db.conversations.create_index(
            [("archived", ASCENDING), ("updated_at", DESCENDING), ("_id", DESCENDING)],
            background=True, name="idx_conversation_list"
        )
        await db.messages.create_index(
            [("conversation_id", ASCENDING), ("seq", ASCENDING)],
            unique=True, background=True, name="idx_message_conv_seq"
        )
        logger.info("MongoDB indexes verified successfully.")
    except Exception as e:
        logger.error(f"Failed to ensure indexes: {e}")
        raise
