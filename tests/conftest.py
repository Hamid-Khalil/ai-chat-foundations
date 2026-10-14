import hashlib
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings (no network, no quota)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16) % 100
            res.append([float(h) / 100.0, 0.5, 0.5, 0.5])
        return res

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class FailingEmbeddings(FakeEmbeddings):
    """Fails on the second batch call to verify transaction rollback."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls >= 2:
            raise ValueError("Simulated embeddings provider failure on batch 2")
        return super().embed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@pytest.fixture
def temp_chroma(tmp_path):
    store = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    return store


@pytest.fixture(autouse=True)
def patch_vector_store(temp_chroma):
    with (
        patch(
            "app.services.ingestion_service.get_vector_store", return_value=temp_chroma
        ),
        patch(
            "app.services.retrieval_service.get_vector_store", return_value=temp_chroma
        ),
    ):
        yield
