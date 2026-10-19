import asyncio
import logging
from app.repositories.indexes import ensure_indexes
from app.core.mongo import init_mongo_client, close_mongo_client, get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    await init_mongo_client()
    db = get_database()
    try:
        await db.command("ping")
        logger.info("MongoDB ping successful for database '%s'.", db.name)
        await ensure_indexes(db)
    finally:
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
