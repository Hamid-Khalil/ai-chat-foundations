"""Task 0.2: First contact with Gemini Embeddings and Chroma vector store."""

import asyncio
import math
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.embeddings import get_embeddings


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Manual cosine similarity calculation: (v1 . v2) / (||v1|| * ||v2||)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


async def main() -> None:
    embeddings = get_embeddings()

    print("=" * 60)
    print("STEP 1: Embedding Vector Lengths")
    print("=" * 60)
    texts = [
        "How many vacation days do employees get?",
        "Employees receive 24 days of paid leave per year.",
        "The office coffee machine is cleaned every Monday.",
    ]
    # Async check
    vecs = await embeddings.aembed_documents(texts)
    for i, v in enumerate(vecs):
        print(f"Text [{i}] vector length: {len(v)}")

    print("\n" + "=" * 60)
    print("STEP 2: Manual Cosine Similarity (Dot Product / Norms)")
    print("=" * 60)
    sim_0_1 = cosine_similarity(vecs[0], vecs[1])
    sim_0_2 = cosine_similarity(vecs[0], vecs[2])
    sim_1_2 = cosine_similarity(vecs[1], vecs[2])
    print(f"Pair 0-1 (Question vs Answer):     {sim_0_1:.4f}")
    print(f"Pair 0-2 (Question vs Unrelated):  {sim_0_2:.4f}")
    print(f"Pair 1-2 (Answer vs Unrelated):    {sim_1_2:.4f}")
    print(f"Closest pair: Pair 0-1 (higher by {sim_0_1 - sim_0_2:.4f})")

    print("\n" + "=" * 60)
    print("STEP 3: embed_query vs embed_documents Task Types")
    print("=" * 60)
    vq = await embeddings.aembed_query(texts[0])
    vd = vecs[0]
    identical = vq == vd
    print(f"Are query and document vectors identical? {identical}")
    if not identical:
        query_doc_sim = cosine_similarity(vq, vd)
        print(f"Similarity between query vector and doc vector: {query_doc_sim:.4f}")
        print(
            "Reason: LangChain/Gemini applies RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT task types."
        )

    print("\n" + "=" * 60)
    print("STEP 4: Chroma Search (hnsw:space = cosine)")
    print("=" * 60)
    smoke_dir = PROJECT_ROOT / "data" / "smoke"
    shutil.rmtree(smoke_dir, ignore_errors=True)

    store_cosine = Chroma(
        collection_name="smoke_test_cosine",
        embedding_function=embeddings,
        persist_directory=str(smoke_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    docs = [
        Document(
            page_content="Full-time employees receive 24 days of paid leave per year.",
            metadata={"doc_id": "d1"},
        ),
        Document(
            page_content="Passphrases must be at least 16 characters with special symbols.",
            metadata={"doc_id": "d2"},
        ),
        Document(
            page_content="The Professional subscription tier grants 25 concurrent projects.",
            metadata={"doc_id": "d3"},
        ),
        Document(
            page_content="External contractors are not eligible for paid leave carryover.",
            metadata={"doc_id": "d4"},
        ),
        Document(
            page_content="Hardware security keys with FIDO2 are strictly required for production.",
            metadata={"doc_id": "d5"},
        ),
        Document(
            page_content="Starter plan costs $49 monthly with 50 GB storage.",
            metadata={"doc_id": "d6"},
        ),
    ]
    await store_cosine.aadd_documents(docs)

    queries = [
        ("Clearly Related", "How many leave days do employees get?"),
        ("Loosely Related", "What is the policy for sick time or absence?"),
        ("Nonsense Query", "What is the capital of Mars?"),
    ]

    print("Chroma Cosine Distance Output (Lower is Closer):")
    for label, q in queries:
        results = await store_cosine.asimilarity_search_with_score(q, k=3)
        print(f"\nQuery ({label}): '{q}'")
        for rank, (doc, dist) in enumerate(results, start=1):
            sim = 1.0 - dist
            print(
                f"  Rank {rank} | Cosine Distance: {dist:.4f} | Inverted Similarity: {sim:.4f} | Content: {doc.page_content[:50]}..."
            )

    print("\n" + "=" * 60)
    print("STEP 5: Chroma Search (Default L2 / Euclidean Distance)")
    print("=" * 60)
    smoke_l2_dir = PROJECT_ROOT / "data" / "smoke_l2"
    shutil.rmtree(smoke_l2_dir, ignore_errors=True)

    store_l2 = Chroma(
        collection_name="smoke_test_l2",
        embedding_function=embeddings,
        persist_directory=str(smoke_l2_dir),
    )
    await store_l2.aadd_documents(docs)

    for label, q in queries:
        results = await store_l2.asimilarity_search_with_score(q, k=1)
        doc, l2_dist = results[0]
        print(
            f"Query ({label}): L2 Distance: {l2_dist:.4f} -> {doc.page_content[:45]}..."
        )

    print("\n" + "=" * 60)
    print("STEP 6: Persistence Verification")
    print("=" * 60)
    # Reopen existing store without adding docs
    persisted_store = Chroma(
        collection_name="smoke_test_cosine",
        embedding_function=embeddings,
        persist_directory=str(smoke_dir),
    )
    persisted_count = len(persisted_store.get()["ids"])
    print(
        f"Re-opened Chroma store without inserting: found {persisted_count} documents."
    )
    assert persisted_count == 6, f"Expected 6 documents, found {persisted_count}"
    print("Persistence verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
