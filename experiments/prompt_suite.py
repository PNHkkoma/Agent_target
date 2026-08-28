from __future__ import annotations

import asyncio
import json
from pathlib import Path

from experiments.common import post, save


CASES = json.loads(
    (Path(__file__).parents[1] / "tests" / "prompt_cases.json").read_text(encoding="utf-8")
)


async def evaluate(case: dict[str, str]) -> dict[str, object]:
    endpoint = "/chat/structured" if case["kind"] == "structured" else "/chat"
    try:
        response = await post(endpoint, {"message": case["prompt"]})
        if case["kind"] == "structured":
            data = response["data"]
            passed = (
                isinstance(data["category"], str)
                and isinstance(data["budget_max"], int)
                and data["budget_max"] > 0
                and isinstance(data["requirements"], list)
                and bool(data["requirements"])
            )
        else:
            passed = isinstance(response["content"], str) and bool(response["content"].strip())
        return {**case, "passed": passed, "response": response}
    except Exception as exc:
        return {**case, "passed": False, "error": str(exc)}


async def main() -> None:
    # Sequential calls make rate-limit behavior and logs easier to inspect.
    results = [await evaluate(case) for case in CASES]
    destination = save("prompt-suite.json", results)
    passed = sum(bool(result["passed"]) for result in results)
    print(f"{passed}/{len(results)} prompts passed; details: {destination}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

