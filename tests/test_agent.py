from __future__ import annotations

import json

import pytest

from app.agent.errors import MaxAgentStepsExceededError
from app.agent.runner import ShoppingAgentV0
from app.config import Settings
from app.llm.router import ModelRouter
from app.schemas.agent import AgentRequest
from app.schemas.chat import FunctionCall, ToolCall
from app.tools import build_default_registry
from tests.fakes import FakeProvider, response


# Nhận ID, tên và JSON arguments; trả ToolCall giả để điều khiển model trong unit test.
def tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments=arguments),
    )


# Nhận provider và ToolCall; trả phản hồi model giả có finish_reason=tool_calls.
def tool_response(provider: str, call: ToolCall):
    result = response(provider, "")
    result.finish_reason = "tool_calls"
    result.tool_calls = [call]
    return result


# Nhận chuỗi response giả và giới hạn; trả ShoppingAgentV0 cùng provider để test trace/messages.
def make_runner(
    responses,
    *,
    agent_max_steps: int = 8,
    agent_max_duplicate_calls: int = 1,
) -> tuple[ShoppingAgentV0, FakeProvider]:
    settings = Settings(
        _env_file=None,
        llm_provider="test",
        llm_fallback_providers=[],
        llm_max_retries=0,
        agent_max_steps=agent_max_steps,
        agent_max_duplicate_calls=agent_max_duplicate_calls,
        tool_timeout_seconds=0.1,
    )
    provider = FakeProvider("test", responses=responses)
    router = ModelRouter({"test": provider}, settings)
    return ShoppingAgentV0(router, build_default_registry(0.1), settings), provider


# Gửi câu hỏi khái niệm; xác nhận agent trả lời ngay và vẫn quảng bá đúng năm tool.
@pytest.mark.asyncio
async def test_agent_knows_when_no_tool_is_needed() -> None:
    runner, provider = make_runner([response("test", "Balo là một loại túi đeo.")])
    result = await runner.run(AgentRequest(message="Balo là gì?"), request_id="agent-1")
    assert result.total_tool_calls == 0
    assert len(provider.last_tools) == 5


# Giả lập model gọi search; xác nhận kết quả catalog được gửi lại trước câu trả lời cuối.
@pytest.mark.asyncio
async def test_agent_executes_product_search() -> None:
    call = tool_call(
        "search", "search_products", '{"category":"backpack","max_price":800000}'
    )
    runner, provider = make_runner(
        [tool_response("test", call), response("test", "Có hai balo phù hợp.")]
    )
    result = await runner.run(
        AgentRequest(message="Tìm balo dưới 800k"), request_id="agent-2"
    )
    assert result.total_tool_calls == 1
    tool_message = next(item for item in provider.last_messages if item.role == "tool")
    assert json.loads(tool_message.content)["data"]["count"] == 2


# Giả lập arguments sai rồi sửa; xác nhận agent nhận INVALID_ARGUMENTS và tiếp tục được.
@pytest.mark.asyncio
async def test_invalid_arguments_are_returned_for_correction() -> None:
    bad = tool_call("bad", "get_product_detail", '{"product_ref":{"id":"P001"}}')
    good = tool_call("good", "get_product_detail", '{"product_ref":"P001"}')
    runner, _ = make_runner(
        [tool_response("test", bad), tool_response("test", good), response("test", "650k")]
    )
    result = await runner.run(
        AgentRequest(message="Giá P001?"), request_id="agent-3"
    )
    tool_results = [step.result for step in result.trace if step.type == "tool"]
    assert tool_results[0]["error"] == "INVALID_ARGUMENTS"
    assert tool_results[1]["status"] == "SUCCESS"


# Giả lập so sánh hai sản phẩm rồi tính ship; xác nhận đúng chuỗi ba tool nhiều bước.
@pytest.mark.asyncio
async def test_multi_step_compare_and_shipping() -> None:
    calls = [
        tool_call("a", "get_product_detail", '{"product_ref":"P001"}'),
        tool_call("b", "get_product_detail", '{"product_ref":"P002"}'),
        tool_call(
            "ship",
            "calculate_shipping_fee",
            '{"product_refs":["P001","P002"],"destination":"Hà Nội"}',
        ),
    ]
    runner, _ = make_runner(
        [*(tool_response("test", call) for call in calls), response("test", "Đã so sánh.")]
    )
    result = await runner.run(
        AgentRequest(message="So A với B và tính ship Hà Nội"), request_id="agent-4"
    )
    assert [step.tool for step in result.trace if step.type == "tool"] == [
        "get_product_detail",
        "get_product_detail",
        "calculate_shipping_fee",
    ]


# Gọi lại cùng tool/arguments; xác nhận lần trùng không được thực thi lần hai.
@pytest.mark.asyncio
async def test_duplicate_tool_call_is_not_executed_twice() -> None:
    first = tool_call("one", "check_inventory", '{"product_ref":"P001"}')
    duplicate = tool_call("two", "check_inventory", '{"product_ref":"P001"}')
    runner, _ = make_runner(
        [tool_response("test", first), tool_response("test", duplicate), response("test", "Còn hàng")]
    )
    result = await runner.run(
        AgentRequest(message="P001 còn hàng không?"), request_id="agent-5"
    )
    tool_results = [step.result for step in result.trace if step.type == "tool"]
    assert tool_results[0]["status"] == "SUCCESS"
    assert tool_results[1]["error"] == "DUPLICATE_TOOL_CALL"


# Ép model gọi tool ngoài allowed_tools; xác nhận registry trả UNKNOWN_TOOL thay vì chạy.
@pytest.mark.asyncio
async def test_request_tool_allowlist_is_enforced() -> None:
    call = tool_call("order", "get_order_status", '{"order_id":"ORD-1001"}')
    runner, _ = make_runner([tool_response("test", call), response("test", "Không thể tra cứu")])
    result = await runner.run(
        AgentRequest(message="Tra đơn ORD-1001", allowed_tools=["search_products"]),
        request_id="agent-6",
    )
    tool_result = next(step.result for step in result.trace if step.type == "tool")
    assert tool_result["error"] == "UNKNOWN_TOOL"


# Giả lập DB lỗi; xác nhận lỗi có cấu trúc được đưa lại model thay vì crash agent loop.
@pytest.mark.asyncio
async def test_tool_failure_is_returned_to_model() -> None:
    call = tool_call("error", "get_order_status", '{"order_id":"ORD-ERROR"}')
    runner, _ = make_runner(
        [tool_response("test", call), response("test", "Không thể xác minh trạng thái.")]
    )
    result = await runner.run(
        AgentRequest(message="Tra đơn ORD-ERROR"), request_id="agent-7"
    )
    tool_result = next(step.result for step in result.trace if step.type == "tool")
    assert tool_result["error"] == "TOOL_EXECUTION_ERROR"
    assert "Không thể" in result.content


# Giới hạn chỉ một model step; xác nhận vòng lặp chưa hoàn tất bị dừng có kiểm soát.
@pytest.mark.asyncio
async def test_max_steps_stops_unfinished_loop() -> None:
    call = tool_call("one", "get_product_detail", '{"product_ref":"P001"}')
    runner, _ = make_runner([tool_response("test", call)], agent_max_steps=1)
    with pytest.raises(MaxAgentStepsExceededError):
        await runner.run(
            AgentRequest(message="Tiếp tục mãi"), request_id="agent-8"
        )
