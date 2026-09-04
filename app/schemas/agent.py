from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.chat import ChatRequest


# Request của Agent, mở rộng ChatRequest bằng allowlist tool tùy chọn.
class AgentRequest(ChatRequest):
    allowed_tools: list[str] | None = Field(default=None, max_length=20)
    include_trace: bool = True

    # Nhận danh sách allowed_tools; trả lại danh sách hợp lệ hoặc báo lỗi nếu bị trùng tên.
    @field_validator("allowed_tools")
    @classmethod
    def tool_names_must_be_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("allowed_tools must not contain duplicates")
        return value


# Một sự kiện model hoặc tool trong trace của agent loop.
class AgentStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step: int = Field(ge=1)
    type: Literal["model", "tool"]
    provider: str | None = None
    model: str | None = None
    tool: str | None = None
    tool_call_id: str | None = Field(default=None, alias="toolCallId")
    arguments: dict[str, Any] | str | None = None
    result: dict[str, Any] | None = None
    latency_ms: int | None = Field(default=None, alias="latencyMs")
    input_tokens: int | None = Field(default=None, alias="inputTokens")
    output_tokens: int | None = Field(default=None, alias="outputTokens")


# Kết quả cuối của Shopping Agent kèm tổng usage và trace hành động.
class AgentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str
    provider: str
    model: str
    total_steps: int = Field(alias="totalSteps")
    total_tool_calls: int = Field(alias="totalToolCalls")
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    latency_ms: int = Field(alias="latencyMs")
    trace: list[AgentStep] = Field(default_factory=list)
