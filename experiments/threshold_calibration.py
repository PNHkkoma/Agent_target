from __future__ import annotations

import json
from pathlib import Path

from experiments.common import save


SOURCE = Path("experiment-results/rag-evaluation.json")


# Nhận một kết quả eval; trả relevance/confidence top-1 để tối ưu rule confidence không dùng score cũ.
def signals(result: dict) -> tuple[float, float]:
    hits = result.get("actual_hits", [])
    relevance = float(hits[0]["score"]) if hits else 0.0
    confidence = float(result.get("confidence") or 0.0)
    return relevance, confidence


# Không nhận đầu vào; tìm ngưỡng high/medium ưu tiên không trả lời trực tiếp khi query không có evidence.
def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    results = report["results"]
    candidates = sorted({signals(result)[0] for result in results})
    confidence_candidates = sorted({signals(result)[1] for result in results})
    choices: list[dict] = []
    for high in candidates:
        for minimum_confidence in confidence_candidates:
            direct = [
                result
                for result in results
                if signals(result)[0] >= high
                and signals(result)[1] >= minimum_confidence
            ]
            unsafe_direct = sum(
                result["expected_document"] is None for result in direct
            )
            correct_direct = sum(
                result["expected_document"] is not None
                and result["hit_at_1"]
                for result in direct
            )
            choices.append(
                {
                    "high_relevance_threshold": high,
                    "min_reranker_confidence": minimum_confidence,
                    "unsafe_direct_answers": unsafe_direct,
                    "correct_direct_answers": correct_direct,
                }
            )
    best = sorted(
        choices,
        key=lambda item: (
            item["unsafe_direct_answers"],
            -item["correct_direct_answers"],
            -item["min_reranker_confidence"],
        ),
    )[0]
    calibration = {
        "source": str(SOURCE),
        "recommended": {
            **best,
            "medium_relevance_threshold": 0.5,
            "rationale": "Medium giữ query có tín hiệu nhưng không đủ high để yêu cầu làm rõ.",
        },
    }
    destination = save("threshold-calibration.json", calibration)
    print(
        "threshold calibration: "
        f"high={best['high_relevance_threshold']} "
        f"minimum_confidence={best['min_reranker_confidence']} "
        f"unsafe_direct={best['unsafe_direct_answers']} "
        f"correct_direct={best['correct_direct_answers']}; details={destination}"
    )


if __name__ == "__main__":
    main()
