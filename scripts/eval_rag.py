"""Automated RAG evaluation harness for Task 2.2."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure app imports work from the scripts directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.ingestion_service import get_ingestion_service
from app.services.retrieval_service import get_retrieval_service


async def reingest_all() -> None:
    """Wipe the collection and re-ingest all sample files."""
    ingest_svc = get_ingestion_service()

    # 1. Delete all existing documents
    docs = await ingest_svc.list_documents()
    for doc in docs.documents:
        await ingest_svc.delete_document(doc.document_id)

    # 2. Find and ingest top-level files in knowledge_samples/
    samples_dir = PROJECT_ROOT / "knowledge_samples"
    for filepath in samples_dir.glob("*"):
        if filepath.is_file() and filepath.name.endswith((".md", ".txt")):
            with open(filepath, "rb") as f:
                data = f.read()
            await ingest_svc.ingest_document(filename=filepath.name, data=data)
    print("✅ Collection wiped and re-ingested with current chunk settings.\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reingest", action="store_true", help="Wipe and re-ingest before eval"
    )
    args = parser.parse_args()

    if args.reingest:
        await reingest_all()

    retrieval_svc = get_retrieval_service()
    settings = get_settings()

    questions_path = PROJECT_ROOT / "tests" / "rag_eval" / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions = qa_data.get("questions", [])

    print(
        f"=== RAG Evaluation (Chunk Size: {settings.rag_chunk_size}, Overlap: {settings.rag_chunk_overlap}) ==="
    )
    print(f"{'Question':<50} | {'Expected Doc':<22} | {'Rank'} | {'Score'}")
    print("-" * 95)

    answerable_scores = []
    unanswerable_scores = []
    hits_at_1 = 0
    hits_at_k = 0

    # Evaluate Answerable Questions
    for q in questions:
        expected = q.get("expected_document")
        if not expected:
            continue

        results = await retrieval_svc.retrieve(
            query=q["question"], top_k=settings.rag_top_k_default, min_score=0.0
        )

        top_score = results[0].score if results else 0.0
        answerable_scores.append(top_score)

        rank_str = "-"
        for r in results:
            if r.filename == expected:
                rank_str = str(r.rank)
                hits_at_k += 1
                if r.rank == 1:
                    hits_at_1 += 1
                break

        short_q = (
            (q["question"][:47] + "...") if len(q["question"]) > 50 else q["question"]
        )
        print(f"{short_q:<50} | {expected:<22} | {rank_str:<4} | {top_score:.4f}")

    print("-" * 95)
    print("Unanswerable Questions Top Scores:")

    # Evaluate Unanswerable Questions
    for q in questions:
        if q.get("expected_document"):
            continue

        results = await retrieval_svc.retrieve(
            query=q["question"], top_k=1, min_score=0.0
        )
        top_score = results[0].score if results else 0.0
        unanswerable_scores.append(top_score)

        short_q = (
            (q["question"][:65] + "...") if len(q["question"]) > 68 else q["question"]
        )
        print(f"- {short_q:<68} -> {top_score:.4f}")

    avg_ans = (
        sum(answerable_scores) / len(answerable_scores) if answerable_scores else 0.0
    )
    avg_unans = (
        sum(unanswerable_scores) / len(unanswerable_scores)
        if unanswerable_scores
        else 0.0
    )
    total_ans = len(answerable_scores)

    print("\n=== Summary ===")
    print(f"Hit@1: {hits_at_1}/{total_ans}")
    print(f"Hit@k: {hits_at_k}/{total_ans}")
    print(f"Avg Top Score (Answerable):   {avg_ans:.4f}")
    print(f"Avg Top Score (Unanswerable): {avg_unans:.4f}")
    print(
        "===============================================================================================\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
