import json
from pathlib import Path

import pytest


CASES = json.loads(
    (Path(__file__).parent / "agent_prompt_cases.json").read_text(encoding="utf-8")
)


# Không nhận đầu vào; xác nhận corpus có đủ 50 case, 5 tool và mọi nhóm rủi ro bắt buộc.
def test_agent_evaluation_has_required_decision_cases() -> None:
    assert len(CASES) == 50
    all_expected = {
        tool
        for case in CASES
        for tool in case.get("expected_tools", [])
    }
    assert all_expected == {
        "search_products",
        "get_product_detail",
        "check_inventory",
        "calculate_shipping_fee",
        "get_order_status",
    }
    categories = {case["category"] for case in CASES}
    assert {
        "no_tool",
        "one_tool",
        "multiple_tools",
        "missing_information",
        "tool_error",
        "argument_invalid",
        "product_not_found",
        "tool_timeout",
        "forbidden_action",
        "forbidden_tool",
    } <= categories


# Nhận từng case từ corpus; xác nhận contract evaluator của case đó đầy đủ và hợp lệ.
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_agent_prompt_case_contract(case) -> None:
    if case.get("kind") == "registry":
        assert case["tool"]
        assert case["expected_error"]
    else:
        assert case["match"] in {"exact", "counts"}
        assert case["prompt"].strip()
