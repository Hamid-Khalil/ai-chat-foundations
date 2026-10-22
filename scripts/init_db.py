import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.mongo import close_mongo_client, get_database, init_mongo_client
from app.repositories.indexes import ensure_indexes


async def main():
    await init_mongo_client()
    db = get_database()
    await ensure_indexes(db)
    print("Database initialization and index creation COMPLETE.")
    await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
