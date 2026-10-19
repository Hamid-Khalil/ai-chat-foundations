import logging
from typing import Optional
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Single module-level client instance, set only by the FastAPI lifespan
_client: Optional[AsyncMongoClient] = None


def get_mongo_client() -> AsyncMongoClient:
    """Return the active AsyncMongoClient or raise if not initialized."""
    if _client is None:
        raise RuntimeError(
            "MongoDB client is not initialized. Ensure it is created inside the lifespan."
        )
    return _client


def get_database() -> AsyncDatabase:
    """Return the configured application database instance."""
    settings = get_settings()
    return get_mongo_client()[settings.mongodb_db]


async def init_mongo_client() -> AsyncMongoClient:
    """Create the single AsyncMongoClient instance with UTC timezone awareness."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    _client = AsyncMongoClient(
        settings.mongodb_uri.get_secret_value(),
        tz_aware=True,
        serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
    )
    return _client


async def close_mongo_client() -> None:
    """Close the global AsyncMongoClient cleanly on shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
