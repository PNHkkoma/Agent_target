import json
from pathlib import Path

import pytest


CASES = json.loads(
    (Path(__file__).parent / "agent_prompt_cases.json").read_text(encoding="utf-8")
)


def test_agent_evaluation_has_required_decision_cases() -> None:
    assert 3 <= len(CASES) <= 20
    all_expected = {tool for case in CASES for tool in case["expected_tools"]}
    assert {"calculate", "search_products", "get_product", "get_weather"} <= all_expected
    assert any(not case["expected_tools"] for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_agent_prompt_case_contract(case) -> None:
    assert case["match"] in {"exact", "contains"}
    assert case["prompt"].strip()

