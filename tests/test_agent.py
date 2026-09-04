from __future__ import annotations

import json

import pytest

from app.agent.errors import MaxAgentStepsExceededError
from app.agent.runner import AgentRunner
from app.config import Settings
from app.llm.router import ModelRouter
from app.schemas.agent import AgentRequest
from app.schemas.chat import FunctionCall, ToolCall
from app.tools import build_default_registry
from tests.fakes import FakeProvider, response


def tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments=arguments),
    )


def tool_response(provider: str, call: ToolCall):
    result = response(provider, "")
    result.finish_reason = "tool_calls"
    result.tool_calls = [call]
    return result


def make_runner(
    responses,
    *,
    agent_max_steps: int = 8,
    agent_max_duplicate_calls: int = 1,
) -> tuple[AgentRunner, FakeProvider]:
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
    return AgentRunner(router, build_default_registry(0.1), settings), provider


@pytest.mark.asyncio
async def test_agent_knows_when_no_tool_is_needed() -> None:
    runner, provider = make_runner([response("test", "A backpack is a type of bag.")])
    result = await runner.run(
        AgentRequest(message="What is a backpack?"), request_id="agent-1"
    )
    assert result.total_tool_calls == 0
    assert result.total_steps == 1
    assert provider.last_tools and len(provider.last_tools) == 4


@pytest.mark.asyncio
async def test_agent_executes_calculator_and_returns_result_to_model() -> None:
    call = tool_call("call-1", "calculate", '{"expression":"38 * 27 + 17"}')
    runner, provider = make_runner(
        [tool_response("test", call), response("test", "1043")]
    )
    result = await runner.run(
        AgentRequest(message="38 × 27 + 17 bằng bao nhiêu?"), request_id="agent-2"
    )
    assert result.content == "1043"
    assert result.total_tool_calls == 1
    tool_message = next(message for message in provider.last_messages if message.role == "tool")
    assert json.loads(tool_message.content)["data"]["result"] == 1043


@pytest.mark.asyncio
async def test_invalid_arguments_are_returned_to_model_for_correction() -> None:
    bad = tool_call("call-bad", "calculate", '{"expression":12,"extra":"banana"}')
    corrected = tool_call("call-good", "calculate", '{"expression":"6 * 7"}')
    runner, provider = make_runner(
        [
            tool_response("test", bad),
            tool_response("test", corrected),
            response("test", "42"),
        ]
    )
    result = await runner.run(
        AgentRequest(message="Calculate 6 * 7"), request_id="agent-3"
    )
    assert result.content == "42"
    tool_results = [step.result for step in result.trace if step.type == "tool"]
    assert tool_results[0]["error"] == "INVALID_ARGUMENTS"
    assert tool_results[1]["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_multi_step_shopping_agent_uses_weather_search_and_detail() -> None:
    weather = tool_call("weather", "get_weather", '{"city":"Da Lat"}')
    search = tool_call(
        "search",
        "search_products",
        '{"category":"jacket","max_price":1000000}',
    )
    detail = tool_call("detail", "get_product", '{"product_id":4}')
    runner, _ = make_runner(
        [
            tool_response("test", weather),
            tool_response("test", search),
            tool_response("test", detail),
            response("test", "Choose Jacket B."),
        ]
    )
    result = await runner.run(
        AgentRequest(
            message="I am going to Da Lat with a 1M VND budget. Choose a light jacket."
        ),
        request_id="agent-4",
    )
    assert result.content == "Choose Jacket B."
    assert [step.tool for step in result.trace if step.type == "tool"] == [
        "get_weather",
        "search_products",
        "get_product",
    ]


@pytest.mark.asyncio
async def test_duplicate_tool_call_is_not_executed_twice() -> None:
    first = tool_call("one", "get_product", '{"product_id":1}')
    duplicate = tool_call("two", "get_product", '{"product_id":1}')
    runner, _ = make_runner(
        [
            tool_response("test", first),
            tool_response("test", duplicate),
            response("test", "Travel Bag A costs 450000 VND."),
        ]
    )
    result = await runner.run(
        AgentRequest(message="What is product 1?"), request_id="agent-5"
    )
    tool_results = [step.result for step in result.trace if step.type == "tool"]
    assert tool_results[0]["status"] == "SUCCESS"
    assert tool_results[1]["error"] == "DUPLICATE_TOOL_CALL"


@pytest.mark.asyncio
async def test_max_steps_stops_an_unfinished_loop() -> None:
    call = tool_call("one", "get_product", '{"product_id":1}')
    runner, _ = make_runner([tool_response("test", call)], agent_max_steps=1)
    with pytest.raises(MaxAgentStepsExceededError):
        await runner.run(
            AgentRequest(message="Keep looking forever"), request_id="agent-6"
        )

