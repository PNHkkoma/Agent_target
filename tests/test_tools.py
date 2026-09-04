from __future__ import annotations

import asyncio

import pytest
from pydantic import Field

from app.tools import build_default_registry
from app.tools.registry import (
    ToolArguments,
    ToolDefinition,
    ToolPermission,
    ToolRegistry,
)


@pytest.mark.asyncio
async def test_calculator_executes_arithmetic_without_eval() -> None:
    registry = build_default_registry()
    result = await registry.execute("calculate", '{"expression":"38 * 27 + 17"}')
    assert result.status == "SUCCESS"
    assert result.data["result"] == 1043


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        '{"expression":"__import__(\\"os\\").system(\\"whoami\\")"}',
        '{"expression":"2 ** 100"}',
        '{"expression":"unknown + 1"}',
    ],
)
async def test_calculator_rejects_arbitrary_execution(arguments: str) -> None:
    registry = build_default_registry()
    result = await registry.execute("calculate", arguments)
    assert result.status == "ERROR"
    assert result.error == "TOOL_EXECUTION_ERROR"


@pytest.mark.asyncio
async def test_invalid_arguments_never_reach_handler() -> None:
    registry = build_default_registry()
    result = await registry.execute("search_products", '{"max_price":"banana"}')
    assert result.status == "ERROR"
    assert result.error == "INVALID_ARGUMENTS"


@pytest.mark.asyncio
async def test_product_search_applies_budget_and_weight_in_code() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "search_products",
        {"category": "travel_bag", "max_price": 700_000, "max_weight": 700},
    )
    assert result.status == "SUCCESS"
    assert [product["name"] for product in result.data["products"]] == ["Travel Bag A"]


def test_normalization_supports_vietnamese_d_stroke() -> None:
    from app.tools.products import _normalize
    from app.tools.weather import _normalize_city

    assert _normalize("túi du lịch") == "travel_bag"
    assert _normalize_city("Đà Lạt") == "da lat"


@pytest.mark.asyncio
async def test_get_product_never_invents_missing_product() -> None:
    registry = build_default_registry()
    result = await registry.execute("get_product", {"product_id": 999})
    assert result.status == "SUCCESS"
    assert result.data == {"found": False, "product_id": 999}


@pytest.mark.asyncio
async def test_permission_is_enforced_before_handler() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "get_product",
        {"product_id": 1},
        allowed_permissions={ToolPermission.COMPUTE},
    )
    assert result.status == "ERROR"
    assert result.error == "PERMISSION_DENIED"


# Arguments rỗng cho tool chậm dùng trong kiểm thử timeout.
class SlowArguments(ToolArguments):
    delay: float = Field(gt=0)


async def slow_tool(delay: float) -> dict[str, bool]:
    await asyncio.sleep(delay)
    return {"finished": True}


@pytest.mark.asyncio
async def test_tool_timeout_becomes_structured_result() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow_tool",
            description="A deliberately slow test tool.",
            input_model=SlowArguments,
            handler=slow_tool,
            timeout_seconds=0.01,
            permission=ToolPermission.COMPUTE,
        )
    )
    result = await registry.execute("slow_tool", {"delay": 0.1})
    assert result.status == "ERROR"
    assert result.error == "TOOL_TIMEOUT"
    assert result.retryable is True
