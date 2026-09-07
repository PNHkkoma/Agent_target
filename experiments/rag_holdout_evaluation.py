from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from experiments.common import post, save


ROOT = Path(__file__).parents[1]
CASES = json.loads(
    (ROOT / "tests" / "rag_retrieval_holdout_cases.json").read_text(encoding="utf-8")
)
DEV_CASES = json.loads(
    (ROOT / "tests" / "rag_retrieval_cases.json").read_text(encoding="utf-8")
) + json.loads(
    (ROOT / "tests" / "rag_retrieval_cases_extended.json").read_text(encoding="utf-8")
)


# Nhận một holdout case; gọi retrieval production và trả rank/Hit@K/MRR, không chỉnh pipeline theo kết quả.
async def evaluate_retrieval(case: dict[str, Any]) -> dict[str, Any]:
    response = await post(
        "/rag/search",
        {"query": case["query"], "top_k": 20, "min_score": -1},
    )
    actual = [hit["documentId"] for hit in response["hits"]]
    expected = case["expected_document"]
    rank = actual.index(expected) + 1 if expected in actual else None
    return {
        **case,
        "actual_documents": actual,
        "rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_5": rank is not None and rank <= 5,
        "reciprocal_rank": 1 / rank if rank else 0.0,
    }


# Nhận no-answer case; gọi answer endpoint và trả true khi hệ thống không tạo fact/citation từ knowledge không liên quan.
async def evaluate_no_answer(case: dict[str, Any]) -> dict[str, Any]:
    response = await post("/rag/ask", {"query": case["query"]})
    safe = (
        response["confidenceLevel"] in {"low", "medium"}
        and response["provider"] == "none"
        and not response["citations"]
    )
    return {**case, "response": response, "safe_no_answer": safe}


# Nhận các retrieval result; trả các metric cuối cùng của holdout độc lập theo toàn bộ và từng nhóm query.
def summarize_retrieval(results: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "cases": len(results),
        "hit_at_1": sum(result["hit_at_1"] for result in results) / len(results),
        "hit_at_5": sum(result["hit_at_5"] for result in results) / len(results),
        "mrr": sum(result["reciprocal_rank"] for result in results) / len(results),
    }


# Không nhận đầu vào; kiểm tra holdout tách dev set, chạy duy nhất một lần validation và lưu report bất biến.
async def main() -> None:
    if len(CASES) < 50:
        raise SystemExit("Holdout evaluation requires at least 50 cases")
    dev_queries = {case["query"] for case in DEV_CASES}
    overlap = [case["id"] for case in CASES if case["query"] in dev_queries]
    if overlap:
        raise SystemExit(f"Holdout queries overlap dev set: {overlap}")

    positive_cases = [case for case in CASES if case["expected_document"] is not None]
    no_answer_cases = [case for case in CASES if case["expected_document"] is None]
    retrieval = [await evaluate_retrieval(case) for case in positive_cases]
    no_answer = [await evaluate_no_answer(case) for case in no_answer_cases]
    categories = sorted({case["category"] for case in retrieval})
    report = {
        "holdout_only": True,
        "cases": len(CASES),
        "positive_cases": len(positive_cases),
        "no_answer_cases": len(no_answer_cases),
        "metrics": summarize_retrieval(retrieval),
        "no_answer_detection": sum(result["safe_no_answer"] for result in no_answer)
        / len(no_answer),
        "metrics_by_category": {
            category: summarize_retrieval(
                [result for result in retrieval if result["category"] == category]
            )
            for category in categories
        },
        "retrieval_results": retrieval,
        "no_answer_results": no_answer,
    }
    destination = save("rag-holdout-evaluation.json", report)
    print(f"RAG holdout: {report['metrics']}; no-answer={report['no_answer_detection']}; details={destination}")


if __name__ == "__main__":
    asyncio.run(main())
