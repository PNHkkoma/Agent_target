from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from experiments.common import post, save


CASES = json.loads(
    (Path("tests") / "rag_answer_cases.json").read_text(encoding="utf-8")
)


# Nhận text tiếng Việt; trả chuỗi chuẩn hóa để kiểm các fact bắt buộc không phụ thuộc dấu/case.
def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    no_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", no_marks.replace("đ", "d")).strip()


# Nhận answer và citation; trả true nếu answer nêu số không xuất hiện trong evidence được trích.
def has_unsupported_number(answer: str, citations: list[dict[str, Any]]) -> bool:
    evidence = normalize(" ".join(citation["excerpt"] for citation in citations))
    # Nhãn citation như [S1] là định danh kỹ thuật, không phải fact số cần evidence.
    answer_without_labels = re.sub(r"\[s\d+\]", "", normalize(answer))
    numbers = re.findall(r"\d+(?:[.,]\d+)?", answer_without_labels)
    return any(number not in evidence for number in numbers)


# Nhận một answer case; gọi RAG API và trả groundedness, citation correctness, gate behavior chi tiết.
async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    response = await post("/rag/ask", {"query": case["query"]})
    answer = normalize(response["answer"])
    citations = response["citations"]
    required = [normalize(term) for term in case["required_terms"]]
    mode_ok = response["confidenceLevel"] == case["mode"]
    if case["mode"] == "high":
        supported = all(term in answer for term in required)
        citation_correct = any(
            citation["documentId"] == case["expected_document"] for citation in citations
        )
        critical_hallucination = has_unsupported_number(response["answer"], citations)
        passed = mode_ok and supported and citation_correct and not critical_hallucination
    else:
        supported = all(term in answer for term in required)
        citation_correct = not citations
        critical_hallucination = response["provider"] != "none" or bool(citations)
        passed = mode_ok and supported and citation_correct and not critical_hallucination
    return {
        **case,
        "response": response,
        "mode_ok": mode_ok,
        "grounded": supported,
        "citation_correct": citation_correct,
        "critical_hallucination": critical_hallucination,
        "passed": passed,
    }


# Không nhận đầu vào; chạy 50 answer case và lưu gate report cho groundedness/citation/hallucination.
async def main() -> None:
    results = [await evaluate_case(case) for case in CASES]
    high = [result for result in results if result["mode"] == "high"]
    guarded = [result for result in results if result["mode"] != "high"]
    metrics = {
        "cases": len(results),
        "high_cases": len(high),
        "guarded_cases": len(guarded),
        "grounded_answer_rate": sum(result["grounded"] for result in high) / len(high),
        "citation_correctness": sum(result["citation_correct"] for result in high) / len(high),
        "critical_hallucination_rate": sum(
            result["critical_hallucination"] for result in high
        )
        / len(high),
        "guarded_behavior_rate": sum(result["passed"] for result in guarded)
        / len(guarded),
        "overall_pass_rate": sum(result["passed"] for result in results) / len(results),
    }
    destination = save("rag-answer-evaluation.json", {"metrics": metrics, "results": results})
    print(f"answer evaluation: {metrics}; details={destination}")


if __name__ == "__main__":
    asyncio.run(main())
