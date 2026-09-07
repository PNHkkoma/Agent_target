from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from experiments.common import post, save


ROOT = Path(__file__).parents[1]
DOCUMENTS = json.loads(
    (ROOT / "knowledge" / "sample_documents.json").read_text(encoding="utf-8")
)
CASES = json.loads(
    (ROOT / "tests" / "rag_retrieval_cases.json").read_text(encoding="utf-8")
) + json.loads(
    (ROOT / "tests" / "rag_retrieval_cases_extended.json").read_text(encoding="utf-8")
)
EVAL_MIN_SCORE = float(os.getenv("RAG_EVAL_MIN_SCORE", "0.15"))


# Nhận document mẫu; gọi ingestion API và trả response để ghi lại số chunk/model embedding.
async def ingest_document(document: dict[str, Any]) -> dict[str, Any]:
    return await post("/rag/documents", document)


# Nhận retrieval case; đo ranking không threshold và quyết định có/không đáp án theo threshold cấu hình.
async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    ranking_response = await post(
        "/rag/search",
        {
            "query": case["query"],
            "top_k": 20,
            "min_score": -1,
            "filters": case.get("filters", {}),
        },
    )
    expected = case["expected_document"]
    actual = [hit["documentId"] for hit in ranking_response["hits"]]
    threshold_actual = [
        hit["documentId"]
        for hit in ranking_response["hits"][:5]
        if hit["score"] >= EVAL_MIN_SCORE
    ]
    rank = actual.index(expected) + 1 if expected is not None and expected in actual else None
    answerability_correct = not threshold_actual if expected is None else expected in threshold_actual
    return {
        **case,
        "category": case.get("category", "regression"),
        "actual_documents": actual,
        "actual_hits": [
            {"document_id": hit["documentId"], "score": hit["score"]}
            for hit in ranking_response["hits"]
        ],
        "confidence": ranking_response.get("confidence"),
        "threshold_documents": threshold_actual,
        "rank": rank,
        "hit_at_1": None if expected is None else rank == 1,
        "hit_at_5": None if expected is None else rank is not None and rank <= 5,
        "reciprocal_rank": None if expected is None else (1 / rank if rank else 0),
        "no_answer_correct": None if expected is not None else not threshold_actual,
        "answerability_correct": answerability_correct,
    }


# Nhận các kết quả cùng tên nhóm; trả metric retrieval/answerability của riêng nhóm đó.
def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [result for result in results if result["expected_document"] is not None]
    negatives = [result for result in results if result["expected_document"] is None]
    return {
        "cases": len(results),
        "positive_cases": len(positives),
        "no_answer_cases": len(negatives),
        "hit_at_1": (
            sum(bool(result["hit_at_1"]) for result in positives) / len(positives)
            if positives
            else None
        ),
        "hit_at_5": (
            sum(bool(result["hit_at_5"]) for result in positives) / len(positives)
            if positives
            else None
        ),
        "mrr": (
            sum(float(result["reciprocal_rank"]) for result in positives) / len(positives)
            if positives
            else None
        ),
        "no_answer_detection": (
            sum(bool(result["no_answer_correct"]) for result in negatives) / len(negatives)
            if negatives
            else None
        ),
        "answerability_accuracy": sum(
            bool(result["answerability_correct"]) for result in results
        )
        / len(results),
    }


# Không nhận đầu vào; ingest corpus, đo 100 query theo nhóm và lưu semantic baseline đầy đủ.
async def main() -> None:
    ingestions = [await ingest_document(document) for document in DOCUMENTS]
    results = [await evaluate_case(case) for case in CASES]
    metrics = summarize(results)
    metrics["embedding_model"] = ingestions[0]["embeddingModel"] if ingestions else None
    metrics["provisional_min_score"] = EVAL_MIN_SCORE
    categories = sorted({result["category"] for result in results})
    by_category = {
        category: summarize([r for r in results if r["category"] == category])
        for category in categories
    }
    destination = save(
        "rag-evaluation.json",
        {
            "metrics": metrics,
            "metrics_by_category": by_category,
            "ingestions": ingestions,
            "results": results,
        },
    )
    print(f"RAG retrieval baseline: {metrics}; details={destination}")
    if metrics["cases"] < 100:
        raise SystemExit("RAG evaluation requires at least 100 cases")


if __name__ == "__main__":
    asyncio.run(main())
