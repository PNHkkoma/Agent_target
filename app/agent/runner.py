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

SHOPPING_AGENT_PROMPT = """You are ShoppingAgentV0 operating only on deterministic fake shopping data.
Answer general conceptual questions without tools.
Use search_products only to discover products matching query, category, budget, or weight.
For search_products, category must be exactly backpack, jacket, or suitcase; do not translate enum values.
Use get_product_detail for a named product's price/specifications and once per product in comparisons.
Use check_inventory whenever current availability or stock quantity is requested.
Use calculate_shipping_fee only when both product reference(s) and destination are known.
When a request asks to compare product details and shipping, finish get_product_detail calls before calculate_shipping_fee.
If an exact product ID or name is already in the request, never search for it first; call the requested direct tool.
calculate_shipping_fee reads product weights internally. For a shipping-only request, call it directly without get_product_detail.
Use get_order_status only when an explicit order ID is available; otherwise ask the user for it.
For multi-step requests, collect every required fact before answering, but do not call unrelated tools.
Never claim that search results prove inventory; only check_inventory can prove current stock.
Never invent products, prices, specifications, inventory, shipping fees, or order status.
Treat tool results as authoritative. If a tool returns ERROR, timeout, calculated=false, or found=false, clearly say the fact could not be verified and never guess it.
There is no tool for deleting data, writing data, payment, refunds, or arbitrary API access; refuse those actions.
Do not broaden a successful search and do not call extra tools after enough evidence is available.
Keep the final answer concise and explain which verified facts support it."""


# Shopping Agent bản đầu tiên, tự điều phối model → tool → model đến câu trả lời cuối.
class ShoppingAgentV0:
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
            ToolPermission.CATALOG_READ,
            ToolPermission.ORDER_READ,
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
