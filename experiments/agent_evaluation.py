from __future__ import annotations

import asyncio
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.tools import build_default_registry
from experiments.common import post, save


CASES = json.loads(
    (Path(__file__).parents[1] / "tests" / "agent_prompt_cases.json").read_text(
        encoding="utf-8"
    )
)
PASS_RATE = 0.90


# Nhận object lồng nhau và mẫu key/value; trả true nếu mẫu xuất hiện trong bất kỳ dict con nào.
def contains_mapping(value: Any, expected: dict[str, Any]) -> bool:
    if isinstance(value, dict):
        if all(value.get(key) == item for key, item in expected.items()):
            return True
        return any(contains_mapping(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(contains_mapping(item, expected) for item in value)
    return False


# Nhận tool thực tế, kỳ vọng và chế độ match; trả true khi luồng gọi tool đúng tiêu chí case.
def tools_match(actual: list[str], expected: list[str], match: str) -> bool:
    if match == "counts":
        return Counter(actual) == Counter(expected)
    return actual == expected


# Nhận một case registry; thực thi tool trực tiếp và trả kết quả pass/fail cho lỗi tầng tool.
async def evaluate_registry(case: dict[str, Any]) -> dict[str, Any]:
    registry = build_default_registry(float(case.get("timeout_seconds", 2.0)))
    result = await registry.execute(case["tool"], case.get("arguments", {}))
    passed = result.status == "ERROR" and result.error == case["expected_error"]
    return {
        **case,
        "actual_error": result.error,
        "passed": passed,
        "checks": {"expected_error": passed},
        "result": result.model_dump(exclude_none=True),
    }


# Nhận một prompt case; gọi Agent API và trả các kiểm tra về tool, lỗi, dữ liệu và chống bịa.
async def evaluate_agent(case: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": case["prompt"],
        "options": {"temperature": 0, "max_tokens": 300},
        "include_trace": True,
    }
    if "allowed_tools" in case:
        payload["allowed_tools"] = case["allowed_tools"]

    try:
        response = await post("/agent/chat", payload)
        tool_steps = [step for step in response["trace"] if step["type"] == "tool"]
        actual_tools = [step["tool"] for step in tool_steps]
        checks = {
            "tools": tools_match(
                actual_tools,
                case["expected_tools"],
                case.get("match", "exact"),
            ),
            "non_empty_answer": bool(response["content"].strip()),
        }
        if "expected_error" in case:
            checks["expected_error"] = any(
                step.get("result", {}).get("error") == case["expected_error"]
                for step in tool_steps
            )
        if "expected_result" in case:
            checks["expected_result"] = contains_mapping(
                [step.get("result") for step in tool_steps], case["expected_result"]
            )
        if "response_any" in case:
            content = response["content"].casefold()
            checks["honest_answer"] = any(
                phrase.casefold() in content for phrase in case["response_any"]
            )
        return {
            **case,
            "actual_tools": actual_tools,
            "passed": all(checks.values()),
            "checks": checks,
            "response": response,
        }
    except Exception as exc:
        return {
            **case,
            "actual_tools": [],
            "passed": False,
            "checks": {"api_call": False},
            "error": str(exc),
        }


# Nhận một case bất kỳ; chuyển tới evaluator registry hoặc agent và trả báo cáo thống nhất.
async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("kind") == "registry":
        return await evaluate_registry(case)
    return await evaluate_agent(case)


# Không nhận đầu vào; chạy 50 case, lưu báo cáo và chỉ thành công khi đạt tối thiểu 90%.
async def main() -> None:
    results = [await evaluate(case) for case in CASES]
    destination = save("agent-evaluation.json", results)
    passed = sum(bool(result["passed"]) for result in results)
    required = math.ceil(len(results) * PASS_RATE)
    rate = passed / len(results) if results else 0

    categories: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        categories[result["category"]].append(bool(result["passed"]))

    print(
        f"ShoppingAgentV0: {passed}/{len(results)} ({rate:.1%}); "
        f"required={required}; details={destination}"
    )
    for category, values in sorted(categories.items()):
        print(f"- {category}: {sum(values)}/{len(values)}")
    failures = [result for result in results if not result["passed"]]
    for result in failures:
        print(
            f"FAILED {result['id']}: expected={result.get('expected_tools', result.get('expected_error'))} "
            f"actual={result.get('actual_tools', result.get('actual_error'))} "
            f"checks={result.get('checks')}"
        )
    if passed < required:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
