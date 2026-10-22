"""FastAPI application setup."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import (
    AppError,
    app_error_handler,
    unhandled_error_handler,
)
from app.core.mongo import close_mongo_client, get_mongo_client, init_mongo_client
from app.repositories.indexes import ensure_indexes
from app.schemas.chat import HealthResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

OPENAPI_TAGS: list[dict[str, Any]] = [
    {
        "name": "Knowledge Base",
        "description": "Document ingestion, listing, and deletion operations.",
    },
    {
        "name": "Retrieval",
        "description": "Raw semantic vector search and score normalization.",
    },
    {
        "name": "RAG Chat",
        "description": "Grounded question answering and conversational RAG generation.",
    },
    {"name": "Assistant", "description": "Agentic tool management."},
    {
        "name": "Tool Chat",
        "description": "Autonomous multi-tool conversational assistant.",
    },
    {"name": "Simple Chat", "description": "Direct single-turn LLM generation."},
    {
        "name": "Conversation Chat",
        "description": "Multi-turn conversational generation.",
    },
    {"name": "Model Settings", "description": "Model parameter configurations."},
    {"name": "Health", "description": "Liveness check."},
]


# Client is created INSIDE the lifespan (never at import time) so that:
# 1. It binds to Uvicorn's active asyncio event loop.
# 2. Importing app.main in unit tests does not open background sockets.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Refuse to start without a valid model key configured."""
    get_settings().require_api_key()

    # Initialize MongoDB Client
    await init_mongo_client()
    try:
        await get_mongo_client().admin.command("ping")
        logging.info("MongoDB connected successfully.")
        await ensure_indexes(get_mongo_client()[get_settings().mongodb_db])
    except Exception as e:
        # Log the host only. NEVER log the URI or password!
        logging.warning(
            "MongoDB ping failed on startup (host: 127.0.0.1:27017). App starting statelessly."
        )

    yield

    # Close MongoDB Client
    await close_mongo_client()


app = FastAPI(
    title="AI Chat Foundations",
    description=(
        "FastAPI service with LangChain and persistent Chroma vector store "
        "supporting grounded Retrieval-Augmented Generation (RAG)."
    ),
    version="1.0.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(api_router, prefix="/api/v1")


@app.get(
    "/api/v1/health",
    tags=["Health"],
    response_model=HealthResponse,
    summary="Health check",
    description="Returns {'status': 'ok'} when the server is operational.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


import time

from pydantic import BaseModel

from app.core.errors import from_db_exception
from app.core.mongo import get_database, get_mongo_client
from app.repositories.indexes import ensure_indexes


class DBHealthResponse(BaseModel):
    status: str
    database: str
    latency_ms: int


@app.get(
    "/api/v1/health/db",
    tags=["Health"],
    response_model=DBHealthResponse,
    summary="Check Database Health",
    responses={503: {"description": "Database unavailable (DB_UNAVAILABLE)"}},
)
async def health_db() -> DBHealthResponse:
    """Pings MongoDB to verify connection health and measures latency."""
    client = get_mongo_client()
    db = get_database()
    start_time = time.perf_counter()

    try:
        await client.admin.command("ping")
        latency = int((time.perf_counter() - start_time) * 1000)
        return DBHealthResponse(status="ok", database=db.name, latency_ms=latency)
    except Exception as e:
        raise from_db_exception(e)
