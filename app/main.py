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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Refuse to start without a valid model key configured."""
    get_settings().require_api_key()
    yield


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
