from __future__ import annotations

import asyncio
import json
from pathlib import Path

from experiments.common import post, save


CASES = json.loads(
    (Path(__file__).parents[1] / "tests" / "agent_prompt_cases.json").read_text(
        encoding="utf-8"
    )
)


# Nhận một test case prompt; gọi API và trả kết quả so sánh chuỗi tool thực tế/kỳ vọng.
async def evaluate(case: dict[str, object]) -> dict[str, object]:
    try:
        response = await post(
            "/agent/chat",
            {
                "message": case["prompt"],
                "options": {"temperature": 0, "max_tokens": 250},
            },
        )
        actual = [
            step["tool"]
            for step in response["trace"]
            if step["type"] == "tool"
        ]
        expected = case["expected_tools"]
        if case["match"] == "exact":
            passed = actual == expected
        else:
            passed = all(tool in actual for tool in expected)
        return {
            **case,
            "actual_tools": actual,
            "passed": passed,
            "response": response,
        }
    except Exception as exc:
        return {**case, "actual_tools": [], "passed": False, "error": str(exc)}


# Không nhận đầu vào; chạy toàn bộ corpus, lưu báo cáo và báo lỗi nếu có case thất bại.
async def main() -> None:
    results = [await evaluate(case) for case in CASES]
    destination = save("agent-evaluation.json", results)
    passed = sum(bool(result["passed"]) for result in results)
    print(f"{passed}/{len(results)} agent cases passed; details: {destination}")
    for result in results:
        print(
            f"- {result['id']}: expected={result['expected_tools']} "
            f"actual={result['actual_tools']} passed={result['passed']}"
        )
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
