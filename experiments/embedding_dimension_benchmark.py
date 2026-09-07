from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.rag.chunking import chunk_document
from app.rag.embeddings import OpenAICompatibleEmbeddingProvider
from app.rag.models import DocumentInput
from app.rag.store import InMemoryVectorStore
from experiments.common import save


ROOT = Path(__file__).parents[1]
DOCUMENTS = json.loads(
    (ROOT / "knowledge" / "sample_documents.json").read_text(encoding="utf-8")
)
CASES = json.loads(
    (ROOT / "tests" / "rag_retrieval_cases.json").read_text(encoding="utf-8")
) + json.loads(
    (ROOT / "tests" / "rag_retrieval_cases_extended.json").read_text(
        encoding="utf-8"
    )
)


# Nhận danh sách kết quả; trả metric ranking chung và theo nhóm để so sánh hai số chiều.
def summarize(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    positives = [item for item in results if item["expected_document"] is not None]
    if not positives:
        return {
            "cases": len(results),
            "positive_cases": 0,
            "hit_at_1": None,
            "hit_at_5": None,
            "mrr": None,
        }
    return {
        "cases": len(results),
        "positive_cases": len(positives),
        "hit_at_1": sum(item["rank"] == 1 for item in positives) / len(positives),
        "hit_at_5": sum(
            item["rank"] is not None and item["rank"] <= 5 for item in positives
        )
        / len(positives),
        "mrr": sum(1 / item["rank"] if item["rank"] else 0 for item in positives)
        / len(positives),
    }


# Nhận số chiều cần benchmark; embedding batch document/query rồi trả ranking của toàn bộ eval set.
async def benchmark_dimension(dimensions: int) -> dict[str, Any]:
    settings = Settings()
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key or settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=dimensions,
        version=f"dimension-benchmark-{dimensions}",
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    store = InMemoryVectorStore(
        embedding_provider=provider.provider_name,
        embedding_model=provider.model_name,
        embedding_dimensions=provider.dimensions,
        embedding_version=provider.version,
    )
    try:
        chunks = []
        documents: list[tuple[DocumentInput, str, list[Any]]] = []
        for raw in DOCUMENTS:
            document = DocumentInput(**raw)
            document_id = document.id or ""
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            document_chunks = chunk_document(
                document,
                document_id=document_id,
                chunk_size_chars=settings.rag_chunk_size_chars,
                overlap_chars=settings.rag_chunk_overlap_chars,
                content_hash=content_hash,
                embedding_provider=provider.provider_name,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimensions,
                embedding_version=provider.version,
            )
            chunks.extend(document_chunks)
            documents.append((document, document_id, document_chunks))
        vectors = await provider.embed([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        for document, document_id, document_chunks in documents:
            await store.upsert_document(document, document_id, document_chunks)

        query_vectors = await provider.embed([case["query"] for case in CASES])
        results: list[dict[str, Any]] = []
        for case, query_vector in zip(CASES, query_vectors):
            hits = await store.search(
                query_vector,
                top_k=20,
                min_score=-1,
                filters=case.get("filters", {}),
            )
            actual = [hit.document_id for hit in hits]
            expected = case["expected_document"]
            rank = actual.index(expected) + 1 if expected in actual else None
            results.append(
                {
                    "id": case["id"],
                    "category": case.get("category", "regression"),
                    "expected_document": expected,
                    "rank": rank,
                    "top_documents": actual[:5],
                }
            )
        categories = sorted({result["category"] for result in results})
        return {
            "dimensions": dimensions,
            "metrics": summarize(results),
            "metrics_by_category": {
                category: summarize(
                    [result for result in results if result["category"] == category]
                )
                for category in categories
            },
            "results": results,
        }
    finally:
        await provider.close()


# Không nhận đầu vào; chạy 384 và full 1536, lưu báo cáo để chọn dimension production.
async def main() -> None:
    report = {
        "model": Settings().embedding_model,
        "benchmarks": [
            await benchmark_dimension(384),
            await benchmark_dimension(1536),
        ],
    }
    destination = save("embedding-dimension-benchmark.json", report)
    for benchmark in report["benchmarks"]:
        print(
            f"dimension={benchmark['dimensions']} metrics={benchmark['metrics']} "
            f"paraphrase={benchmark['metrics_by_category']['paraphrase']} "
            f"hard_negative={benchmark['metrics_by_category']['hard_negative']}"
        )
    print(f"details={destination}")


if __name__ == "__main__":
    asyncio.run(main())
