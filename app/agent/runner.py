from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.agent.errors import (
    AgentConfigurationError,
    AgentTimeoutError,
    DuplicateToolLoopError,
    MaxAgentStepsExceededError,
    MaxToolCallsExceededError,
)
from app.config import Settings
from app.llm.router import ModelRouter
from app.schemas.agent import AgentRequest, AgentResponse, AgentStep
from app.schemas.chat import Message
from app.tools.registry import ToolPermission, ToolRegistry, ToolResult

logger = logging.getLogger("agent_lab.agent")

SHOPPING_AGENT_PROMPT = """You are Shopping Agent V0 operating on deterministic fake data.
You can answer general conceptual questions without tools.
Always call calculate for arithmetic; never calculate mentally.
Use search_products whenever catalog availability, price, weight, or matching products are required.
For search_products, category must be exactly travel_bag or jacket; never translate these enum values.
Use get_product when a specific product ID/detail or comparison requires authoritative details.
After search_products, call get_product for viable candidates before making a final recommendation.
Use get_weather when weather is needed to choose a product.
For multi-step requests, collect all required tool facts before recommending.
Never invent a product, price, specification, inventory status, weather value, or calculation.
Treat tool results as authoritative. If a tool returns ERROR or found=false, explain the limitation instead of guessing.
Do not broaden a successful search and do not call extra tools after you have enough evidence.
Weather data is simulated and must be described as simulated.
Keep the final answer concise and explain the evidence behind a recommendation."""


# Tự điều phối model → tool → model đến khi có câu trả lời cuối.
class AgentRunner:
    # Nhận router, registry và settings; lưu các phụ thuộc cần thiết cho một agent runner.
    def __init__(
        self,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        settings: Settings,
    ) -> None:
        self.model_router = model_router
        self.tool_registry = tool_registry
        self.settings = settings

    # Nhận request agent và request ID; trả kết quả cuối, đồng thời giới hạn tổng thời gian chạy.
    async def run(self, request: AgentRequest, *, request_id: str) -> AgentResponse:
        try:
            async with asyncio.timeout(self.settings.agent_timeout_seconds):
                return await self._run_loop(request, request_id=request_id)
        except TimeoutError as exc:
            raise AgentTimeoutError("Agent exceeded its total timeout") from exc

    # Nhận request và ID; lặp gọi model/tool rồi trả AgentResponse khi model đã đủ dữ kiện.
    async def _run_loop(
        self, request: AgentRequest, *, request_id: str
    ) -> AgentResponse:
        started = time.perf_counter()
        allowed_tools = (
            set(request.allowed_tools)
            if request.allowed_tools is not None
            else set(self.tool_registry.names)
        )
        unknown = allowed_tools.difference(self.tool_registry.names)
        if unknown:
            raise AgentConfigurationError(
                f"Unknown allowed_tools: {', '.join(sorted(unknown))}"
            )

        messages = [
            Message(
                role="system",
                content=f"{SHOPPING_AGENT_PROMPT}\n\nUser instructions:\n{request.system_prompt}",
            ),
            *(
                Message(role=item.role, content=item.content)
                for item in request.history
            ),
            Message(role="user", content=request.message),
        ]
        schemas = self.tool_registry.schemas(allowed_tools)
        permissions = {
            ToolPermission.COMPUTE,
            ToolPermission.CATALOG_READ,
            ToolPermission.WEATHER_READ,
        }
        trace: list[AgentStep] = []
        signatures: dict[str, int] = {}
        total_tool_calls = 0
        input_tokens = 0
        output_tokens = 0

        for step_number in range(1, self.settings.agent_max_steps + 1):
            response, _ = await self.model_router.chat(
                messages,
                request.options,
                task=request.task,
                request_id=request_id,
                tools=schemas,
                tool_choice="auto",
            )
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            trace.append(
                AgentStep(
                    step=step_number,
                    type="model",
                    provider=response.provider,
                    model=response.model,
                    latencyMs=response.latency_ms,
                    inputTokens=response.input_tokens,
                    outputTokens=response.output_tokens,
                )
            )
            self._log_model_step(request_id, step_number, response)

            messages.append(
                Message(
                    role="assistant",
                    content=response.content or None,
                    tool_calls=response.tool_calls or None,
                )
            )
            if not response.tool_calls:
                return AgentResponse(
                    content=response.content,
                    provider=response.provider,
                    model=response.model,
                    totalSteps=step_number,
                    totalToolCalls=total_tool_calls,
                    inputTokens=input_tokens,
                    outputTokens=output_tokens,
                    latencyMs=round((time.perf_counter() - started) * 1000),
                    trace=trace if request.include_trace else [],
                )

            for call in response.tool_calls:
                total_tool_calls += 1
                if total_tool_calls > self.settings.agent_max_tool_calls:
                    raise MaxToolCallsExceededError(
                        "Agent exceeded maximum tool calls per request"
                    )

                decoded_arguments = self._decode_arguments(call.function.arguments)
                signature = self._signature(call.function.name, decoded_arguments)
                duplicate_count = signatures.get(signature, 0)
                signatures[signature] = duplicate_count + 1
                if duplicate_count > self.settings.agent_max_duplicate_calls:
                    raise DuplicateToolLoopError(
                        f"Repeated tool call detected: {call.function.name}"
                    )
                if duplicate_count:
                    result = ToolResult.failure(
                        "DUPLICATE_TOOL_CALL",
                        "This exact tool call was already executed; use the existing result or change arguments.",
                    )
                else:
                    result = await self.tool_registry.execute(
                        call.function.name,
                        call.function.arguments,
                        allowed_tools=allowed_tools,
                        allowed_permissions=permissions,
                    )

                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=result.model_dump_json(),
                    )
                )
                trace.append(
                    AgentStep(
                        step=step_number,
                        type="tool",
                        tool=call.function.name,
                        toolCallId=call.id,
                        arguments=decoded_arguments,
                        result=result.model_dump(exclude_none=True),
                    )
                )
                self._log_tool_step(
                    request_id,
                    step_number,
                    call.function.name,
                    call.id,
                    decoded_arguments,
                    result,
                )

        raise MaxAgentStepsExceededError("Agent exceeded maximum model steps")

    # Nhận arguments JSON từ model; trả object đã giải mã hoặc chuỗi gốc nếu JSON sai.
    @staticmethod
    def _decode_arguments(raw_arguments: str) -> dict[str, Any] | str:
        try:
            decoded = json.loads(raw_arguments)
            return decoded if isinstance(decoded, dict) else raw_arguments
        except json.JSONDecodeError:
            return raw_arguments

    # Nhận tên tool và arguments; trả chữ ký ổn định để phát hiện lời gọi trùng lặp.
    @staticmethod
    def _signature(name: str, arguments: dict[str, Any] | str) -> str:
        normalized = (
            json.dumps(arguments, sort_keys=True, separators=(",", ":"))
            if isinstance(arguments, dict)
            else arguments
        )
        return f"{name}:{normalized}"

    # Nhận ID, số bước và phản hồi model; ghi log quan sát, không trả dữ liệu.
    @staticmethod
    def _log_model_step(request_id: str, step: int, response: Any) -> None:
        logger.info(
            "Agent model step",
            extra={
                "requestId": request_id,
                "event": "MODEL",
                "step": step,
                "provider": response.provider,
                "model": response.model,
                "latencyMs": response.latency_ms,
                "inputTokens": response.input_tokens,
                "outputTokens": response.output_tokens,
                "status": "SUCCESS",
            },
        )

    # Nhận thông tin lần gọi tool và kết quả; ghi trace log, không trả dữ liệu.
    @staticmethod
    def _log_tool_step(
        request_id: str,
        step: int,
        tool: str,
        tool_call_id: str,
        arguments: dict[str, Any] | str,
        result: ToolResult,
    ) -> None:
        logger.info(
            "Agent tool step",
            extra={
                "requestId": request_id,
                "event": "TOOL",
                "step": step,
                "tool": tool,
                "toolCallId": tool_call_id,
                "arguments": arguments,
                "result": result.model_dump(exclude_none=True),
                "status": result.status,
            },
        )
