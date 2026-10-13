"""Embedding model factory."""

import os
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Build and cache the Gemini embeddings instance.

    Uses settings.embedding_model (gemini-embedding-001).
    Task type and output_dimensionality are omitted per Week 8 specifications.
    """
    settings = get_settings()
    api_key = (
        settings.google_api_key.get_secret_value().strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=api_key,
    )
