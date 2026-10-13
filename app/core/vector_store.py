"""Chroma vector store factory."""

import os
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.core.embeddings import get_embeddings

# RULE: One embedding model per collection.
# If EMBEDDING_MODEL is ever changed in .env, data/chroma must be deleted
# and all documents must be re-ingested.


@lru_cache
def get_vector_store() -> Chroma:
    """Return the cached Chroma vector store instance."""
    settings = get_settings()
    os.makedirs(settings.vector_store_dir, exist_ok=True)

    client_settings = ChromaSettings(anonymized_telemetry=False)

    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=settings.vector_store_dir,
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=client_settings,
    )
