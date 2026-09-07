from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


# Các quyền tối thiểu mà một tool được phép yêu cầu.
class ToolPermission(str, Enum):
    CATALOG_READ = "catalog:read"
    ORDER_READ = "order:read"


# Lớp cơ sở cấm arguments dư trước khi gọi business handler.
class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Kết quả tool thống nhất để gửi ngược lại model.
class ToolResult(BaseModel):
    status: str
    data: Any | None = None
    error: str | None = None
    message: str | None = None
    retryable: bool = False

    # Nhận dữ liệu handler trả về và đóng gói thành kết quả SUCCESS cho model.
    @classmethod
    def success(cls, data: Any) -> "ToolResult":
        return cls(status="SUCCESS", data=data)

    # Nhận mã lỗi, thông báo và cờ thử lại; trả về lỗi chuẩn hóa để model xử lý.
    @classmethod
    def failure(
        cls, error: str, message: str, *, retryable: bool = False
    ) -> "ToolResult":
        return cls(
            status="ERROR",
            error=error,
            message=message,
            retryable=retryable,
        )


# Khai báo đầy đủ contract, handler, timeout và permission của một tool.
class ToolDefinition:
    # Nhận metadata, schema đầu vào, handler, timeout và quyền; khởi tạo một tool hoàn chỉnh.
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[ToolArguments],
        handler: Callable[..., Any],
        timeout_seconds: float,
        permission: ToolPermission,
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler
        self.timeout_seconds = timeout_seconds
        self.permission = permission

    # Không nhận dữ liệu ngoài đối tượng hiện tại; trả schema function để gửi cho LLM.
    def to_llm_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


# Quản lý allowlist tool và thực thi arguments đã được kiểm tra.
class ToolRegistry:
    # Không nhận đầu vào; tạo registry rỗng dùng để đăng ký các tool được phép chạy.
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # Nhận một ToolDefinition; đăng ký vào allowlist và không trả về dữ liệu.
    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    # Không nhận đầu vào; trả danh sách tên tool hiện đã đăng ký.
    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # Nhận allowlist tùy chọn; trả danh sách schema của các tool được phép gửi tới LLM.
    def schemas(self, allowed_tools: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            tool.to_llm_schema()
            for name, tool in self._tools.items()
            if allowed_tools is None or name in allowed_tools
        ]

    # Nhận tên tool, arguments JSON và phạm vi quyền; trả ToolResult thành công hoặc lỗi có cấu trúc.
    async def execute(
        self,
        name: str,
        raw_arguments: str | dict[str, Any],
        *,
        allowed_tools: set[str] | None = None,
        allowed_permissions: set[ToolPermission] | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None or (allowed_tools is not None and name not in allowed_tools):
            return ToolResult.failure(
                "UNKNOWN_TOOL", f"Tool '{name}' is not available."
            )
        if allowed_permissions is not None and tool.permission not in allowed_permissions:
            return ToolResult.failure(
                "PERMISSION_DENIED", f"Permission denied for tool '{name}'."
            )
        try:
            decoded = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(decoded, dict):
                raise ValueError("arguments must be a JSON object")
            arguments = tool.input_model.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return ToolResult.failure("INVALID_ARGUMENTS", str(exc))

        # Dùng arguments đã validate để gọi handler sync/async; trả dữ liệu nghiệp vụ thô.
        async def invoke() -> Any:
            values = arguments.model_dump()
            if inspect.iscoroutinefunction(tool.handler):
                return await tool.handler(**values)
            return await asyncio.to_thread(tool.handler, **values)

        try:
            data = await asyncio.wait_for(invoke(), timeout=tool.timeout_seconds)
            return ToolResult.success(data)
        except TimeoutError:
            return ToolResult.failure(
                "TOOL_TIMEOUT",
                f"Tool '{name}' timed out.",
                retryable=True,
            )
        except Exception as exc:
            return ToolResult.failure(
                "TOOL_EXECUTION_ERROR",
                f"Tool '{name}' failed: {type(exc).__name__}",
                retryable=False,
            )
